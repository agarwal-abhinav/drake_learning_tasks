from .base_controller import BaseControllerLeafSystem

from enum import Enum

from pydrake.all import (
    AbstractValue, 
    RigidTransform, 
    PiecewisePolynomial, 
    PiecewisePose, 
)

class PlannerState(Enum): 
    MOVE_TO_CENTER = 0 
    MOVE_BACK_TO_BOX = 1
    TASK_COMPLETE = 2

class EEPickPlaceTrajController(BaseControllerLeafSystem): 
    def __init__(self, 
                 root_cfg, 
                 X_WO): 
        super().__init__(root_cfg)

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )

        self.DeclareAbstractOutputPort(
            "X_WG", 
            lambda: AbstractValue.Make([RigidTransform()]),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        self.X_WO = X_WO 

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.MOVE_TO_CENTER)
        )

        self.execution_start_time = -1 
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.001, 0.0, self.Update)

    def Update(self, context, state)