import numpy as np
import numpy.typing as npt
from enum import Enum
from copy import copy
import logging

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.all import InverseKinematics
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve
from pydrake.all import (
    GcsTrajectoryOptimization,
    HPolyhedron,
    PiecewisePolynomial,
    Point,
    Toppra,
    PathParameterizedTrajectory,
    LeafSystem,
    AbstractValue,
    InputPortIndex,
    RigidTransform,
    MultibodyPlant,
    AbstractValue,
    LeafSystem,
)

logger = logging.getLogger(__name__)


class IiwaPlannerMode(Enum):
    PLAN_GO_PUSH_START = 0
    GO_PUSH_START = 1
    WAIT_PUSH = 2
    DIFF_IK = 3

class IiwaPlanner(LeafSystem):
    """Planner that manages the iiwa going to the start position, waiting and then pushing according to the desired planar position source."""

    def __init__(
        self,
        robot_plant: MultibodyPlant,
        desired_pose: RigidTransform,
        frame_name: str,
        nominal_joint_positions: npt.NDArray[np.float64],
        toppra_vel_lim: float,
        toppra_acc_lim: float,
        initial_delay: float=2.0,
        wait_diff_ik_delay: float=2.0,
    ):
        LeafSystem.__init__(self)
        self._initial_delay = initial_delay
        self._wait_diff_ik_delay = wait_diff_ik_delay
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(IiwaPlannerMode.PLAN_GO_PUSH_START)
        )

        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": initial_delay})
        )

        # For resets
        self._prev_last_reset_time = 0.0
        self._last_reset_time_index = self.DeclareVectorInputPort(
            "last_reset_time", 1
        ).get_index()
        self._reset_position_index = self.DeclareVectorInputPort(
            "reset_position", 7
        ).get_index()

        # For GoPushStart mode:
        num_positions = robot_plant.num_positions()
        self._iiwa_position_measured_index = self.DeclareVectorInputPort(
            "iiwa_position_measured", robot_plant.num_positions()
        ).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )

        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", num_positions, self.CalcIiwaPosition
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.005, 0.0, self.Update)
        # This should work but gives weird bug
        # self.DeclarePerStepDiscreteUpdateEvent(self.Update)

        self._internal_model = robot_plant
        self._desired_pose = desired_pose
        self._frame_name = frame_name
        self._nominal_joint_positions = nominal_joint_positions
        self.trajectory_length = None
        self.toppra_vel_lim = toppra_vel_lim
        self.toppra_acc_lim = toppra_acc_lim

    def Update(self, context, state):
        # FSM Logic for planner
        mode = context.get_abstract_state(self._mode_index).get_value()
        current_time = context.get_time()
        times = context.get_abstract_state(self._times_index).get_value()

        if mode == IiwaPlannerMode.PLAN_GO_PUSH_START:
            if context.get_time() > times["initial"]:
                self.PlanGoPushStart(context, state)
            return
        elif mode == IiwaPlannerMode.GO_PUSH_START:
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()

            if current_time > times["go_push_start_final"]:
                # We have reached the end of the GoPushStart trajectory.
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    IiwaPlannerMode.WAIT_PUSH
                )
                # Debug output
                logger.debug(f"Switching to WAIT_PUSH mode at time {current_time}.")
                current_pos = self.get_input_port(
                    self._iiwa_position_measured_index
                ).Eval(context)
                logger.debug(f"Current position: {current_pos}")
            return
        elif mode == IiwaPlannerMode.WAIT_PUSH:
            if current_time > times["wait_push_final"]:
                # We have reached the end of the GoPushStart trajectory.
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    IiwaPlannerMode.DIFF_IK
                )
                # Debug output
                logger.debug(f"Switching to DIFF_IK mode at time {current_time}.")
                current_pos = self.get_input_port(
                    self._iiwa_position_measured_index
                ).Eval(context)
                logger.debug(f"Current position: {current_pos}")
            return
        # TODO(Adam): add logic to switch back into PLAN_GO_PUSH_START mode upon reset
        elif mode == IiwaPlannerMode.DIFF_IK:
            last_reset_time = self.get_input_port(
                self._last_reset_time_index
            ).Eval(context)
            if self._prev_last_reset_time != last_reset_time:
                self._prev_last_reset_time = last_reset_time
                self.ResetPlanner(context, state)
                # Debug output
                logger.debug(f"Switching to PLAN_GO_PUSH_START mode at time {current_time}.")


    def PlanGoPushStart(self, context, state):
        logger.debug(f"PlanGoPushStart at time {context.get_time()}.")
        q_start = copy(context.get_discrete_state(self._q0_index).get_value())
        q_goal = solve_ik(
            plant=self._internal_model,
            pose=self._desired_pose,
            frame_name="iiwa_link_7",
            default_joint_positions=self._nominal_joint_positions,
            eps = 0.0
        )

        q_traj = self.create_go_push_start_traj(q_goal, q_start)
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(q_traj)
        times = state.get_mutable_abstract_state(int(self._times_index)).get_value()

        times["go_push_start_initial"] = context.get_time()
        times["go_push_start_final"] = q_traj.end_time() + context.get_time()
        times["wait_push_final"] = times["go_push_start_final"] + self._wait_diff_ik_delay
        
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            IiwaPlannerMode.GO_PUSH_START
        )
        self.push_start_pos = q_goal
        self.trajectory_length = q_traj.end_time() + self._wait_diff_ik_delay

    def ResetPlanner(self, context, state):
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            IiwaPlannerMode.PLAN_GO_PUSH_START
        )
        times = {'initial': context.get_time() + self._initial_delay}
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)
        context.get_discrete_state(self._q0_index).set_value(
            self.get_input_port(int(self._reset_position_index)).Eval(context),
        )

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.DIFF_IK:
            output.set_value(InputPortIndex(1))  # DiffIK
        else:
            output.set_value(InputPortIndex(2))  # Wait/GoPushStart

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.DIFF_IK:
            output.set_value(False)  # Pushing (DiffIK)
        else:
            output.set_value(True)  # Wait/GoPushStart

    def CalcIiwaPosition(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PLAN_GO_PUSH_START:
            # Stay in the same position
            q_start = copy(context.get_discrete_state(self._q0_index).get_value())
            output.SetFromVector(q_start)
        elif mode == IiwaPlannerMode.GO_PUSH_START:
            # follow the trajectory
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()

            times = context.get_mutable_abstract_state(
                int(self._times_index)
            ).get_value()

            traj_curr_time = context.get_time() - times["go_push_start_initial"]

            output.SetFromVector(traj_q.value(traj_curr_time))
        elif mode == IiwaPlannerMode.WAIT_PUSH:
            # Stay in the final position of the initial trajectory
            output.SetFromVector(self.push_start_pos)
        elif mode == IiwaPlannerMode.DIFF_IK:
            output.SetFromVector(self.push_start_pos)
            # assert (
            #     False
            # ), "Planner CalcIiwaPosition should not be called in DIFF_IK mode."
        else:
            assert False, "Invalid mode."

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_measured_index)).Eval(context),
        )

    @staticmethod
    def make_traj_toppra(traj, plant, vel_limits, accel_limits, num_grid_points=1000):
        toppra = Toppra(
            traj,
            plant,
            np.linspace(traj.start_time(), traj.end_time(), num_grid_points),
        )
        toppra.AddJointVelocityLimit(-vel_limits, vel_limits)
        toppra.AddJointAccelerationLimit(-accel_limits, accel_limits)
        time_traj = toppra.SolvePathParameterization()
        return PathParameterizedTrajectory(traj, time_traj)

    def create_go_push_start_traj(self, q_goal, q_start):
        plant = self._internal_model
        num_positions = plant.num_positions()

        gcs = GcsTrajectoryOptimization(plant.num_positions())

        workspace = gcs.AddRegions(
            [
                HPolyhedron.MakeBox(
                    plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
                )
            ],
            5,
            1,
            60,
        )

        logger.debug(f"q_start = {q_start}")
        logger.debug(f"q_goal = {q_goal}")

        vel_limits = self.toppra_vel_lim * np.ones(7)  # 0.15
        accel_limits = self.toppra_acc_lim * np.ones(7)
        # Set non-zero h_min for start and goal to enforce zero velocity.
        start = gcs.AddRegions([Point(q_start)], order=1, h_min=0.1)
        goal = gcs.AddRegions([Point(q_goal)], order=1, h_min=0.1)
        goal.AddVelocityBounds([0] * num_positions, [0] * num_positions)
        gcs.AddEdges(start, workspace)
        gcs.AddEdges(workspace, goal)
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        gcs.AddVelocityBounds(-vel_limits, vel_limits)

        traj, result = gcs.SolvePath(start, goal)
        assert(result.is_success())

        traj_toppra = IiwaPlanner.make_traj_toppra(
            traj, 
            plant, 
            vel_limits=vel_limits, 
            accel_limits=accel_limits,
        )

        return traj_toppra
    
    def get_trajectory_length(self):
        return self.trajectory_length

def solve_ik(
    plant: MultibodyPlant,
    pose: RigidTransform,
    frame_name: str,
    default_joint_positions: npt.NDArray[np.float64],
    eps=1e-3
) -> npt.NDArray[np.float64]:
    # Plant needs to be just the robot without other objects
    # Need to create a new context that the IK can use for solving the problem

    ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
    pusher_frame = plant.GetFrameByName(frame_name)
    EPS = eps

    ik.AddPositionConstraint(
        pusher_frame,
        np.zeros(3),
        plant.GetFrameByName("iiwa_link_0"),
        pose.translation() - np.ones(3) * EPS,
        pose.translation() + np.ones(3) * EPS,
    )

    ik.AddOrientationConstraint(
        pusher_frame,
        RotationMatrix(),
        plant.GetFrameByName("iiwa_link_0"),
        pose.rotation(),
        EPS,
    )

    # Cost on deviation from default joint positions
    prog = ik.get_mutable_prog()
    q = ik.q()

    q0 = default_joint_positions
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)

    result = Solve(ik.prog())
    assert result.is_success()

    q_sol = result.GetSolution(q)
    return q_sol
