from evaluators.base_evaluator import BaseEvaluator
from tasks.kuka_pusher_long_context import KukaPlanarPusherLongContextBlock

from utils.file_utils import list_files_in_directory
from utils.logging_utils import IterTee

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import time
import sys
from multiprocessing import Process, Queue 
from queue import Empty 

class KukaPlanarPusherLongContextBlockEvaluator(BaseEvaluator): 
    def __init__(self, root_cfg: DictConfig): 
        super().__init__(root_cfg)

        self.min_seed = self.cfg.min_seed 
        self.max_seed = self.cfg.max_seed
        self.seeds = list(range(self.min_seed, self.max_seed))

        self.checkpoint_list = list_files_in_directory(self.cfg.checkpoint_directory)
        self.num_checkpoints = len(self.checkpoint_list)

        file_names = [os.path.basename(path) for path in self.checkpoint_list]
        meta_output_dir = HydraConfig.get().runtime.output_dir
        self.checkpoint_output_dirs = [os.path.join(meta_output_dir, file_names[i].replace(".ckpt", "")) for i in range(self.num_checkpoints)]

        self.num_processes = self.cfg.num_processes 

    def run_eval(self) -> None: 
        def eval_single_checkpoint(checkpoint_path: str, 
                                   output_dir: str,
                                   seeds: list[int],
                                   process_id: int) -> None:
            
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n\nStarting evaluation for checkpoint: {checkpoint_path}\n using process id: {process_id}\n\n")
            global_log_path = os.path.join(output_dir, "eval_log.txt")
            if os.path.exists(global_log_path): 
                global_log = open(global_log_path, "a")
                global_log.write(f"\n\n=== Appending new eval run started: {time.asctime()} ===\n")
            else: 
                global_log = open(global_log_path, "w")
                global_log.write(f"=== Eval run started: {time.asctime()} ===\n")
            global_log.flush()

            tee = IterTee(sys.stdout, global_log)
            sys.stdout = tee 

            m = 0 
            total_success = 0
            total_mild_success = 0 
            total_return_to_box = 0 
            total_mild_return_to_box = 0 

            total_mid_area = 0 
            total_final_area = 0 
            num_mid_area = 0 
            num_final_area = 0 

            while m < len(seeds): 
                print(self.root_cfg.controller)
                self.root_cfg.controller.checkpoint_path =  checkpoint_path
                task: KukaPlanarPusherLongContextBlock = self.task_class(root_cfg=self.root_cfg)
                controller = self.controller_class(root_cfg=self.root_cfg)

                task.controller = controller 

                os.mkdir(os.path.join(output_dir, f"eval_seed_{seeds[m]}"))
                iter_log_path = os.path.join(output_dir, f"eval_seed_{seeds[m]}", "eval_log.txt")
                iter_log = open(iter_log_path, "w")
                tee.set_iter_file(iter_log)

                if m < self.cfg.save_html_first: 
                    save_html = True 
                else: 
                    save_html = False 
                
                print(f"\n--- Starting eval for seed {seeds[m]}---\n")
                task.reset_robot(seeds[m])

                areas, successes, correct_returns = task.diffusion_rollout(
                    self.cfg.eval_max_time, 
                    save_path = os.path.join(output_dir, f"eval_seed_{seeds[m]}"), 
                    save_html=save_html
                )

                if areas[0] is not None: 
                    total_mid_area += areas[0]
                    num_mid_area += 1
                if areas[1] is not None: 
                    total_final_area += areas[1]
                    num_final_area += 1 
                
                if successes[0] == True: 
                    total_success += 1
                if successes[1] == True: 
                    total_mild_success += 1
                
                if correct_returns[0] == True: 
                    total_return_to_box += 1
                if correct_returns[1] == True: 
                    total_mild_return_to_box += 1

                iter_log.close()
                tee.set_iter_file(None)

                m+=1 

                del task.meshcat
                del task 
                del controller 
            
            print("\n================ Evaluation Summary ================\n")
            print(f"Evaluation for: {self.root_cfg.task.initial_location_type} and seed: {seeds[0]} to {seeds[-1]}")
            print(f"Average mid overlap area: {total_mid_area/num_mid_area if num_mid_area > 0 else 0}")
            print(f"Average final overlap area: {total_final_area/num_final_area if num_final_area > 0 else 0}")
            print(f"Total success count: {total_success} out of {len(seeds)}")
            print(f"Total mild success count: {total_mild_success} out of {len(seeds)}")
            print(f"Total return to a box: {total_return_to_box} out of {len(seeds)}")
            print(f"Total mild return to a box: {total_mild_return_to_box} out of {len(seeds)}")
            print("\n====================================================\n")

            global_log.close()

            return total_success, total_mild_success, total_return_to_box, total_mild_return_to_box

        out_q = Queue() 
        active = [] 
        results = {}
        jobs = list(range(self.num_checkpoints))
        total = len(jobs)
        next_job = 0 

        while next_job < total or active: 
            while next_job < total and len(active) < self.num_processes: 
                p = Process(
                    target = eval_single_checkpoint, 
                    args = (
                        self.checkpoint_list[next_job], 
                        self.checkpoint_output_dirs[next_job],
                        self.seeds,
                        next_job
                    )
                )
                p.start()
                active.append(p)
                next_job += 1

            try: 
                while True: 
                    job_id, res = out_q.get_nowait()
                    results[job_id] = res
            except Empty: 
                pass

            active = [p for p in active if p.is_alive()]

            time.sleep(0.05)

        while len(results) < total: 
            job_id, res = out_q.get()
            results[job_id] = res

        max_success_job_id = -1 
        max_mild_success_job_id = -1 
        max_return_job_id = -1 
        max_mild_return_job_id = -1

        for key in results.keys(): 
            if max_success_job_id == -1 or results[key][0] > results[max_success_job_id][0]: 
                max_success_job_id = key 
            if max_mild_success_job_id == -1 or results[key][1] > results[max_mild_success_job_id][1]: 
                max_mild_success_job_id = key 
            if max_return_job_id == -1 or results[key][2] > results[max_return_job_id][2]: 
                max_return_job_id = key 
            if max_mild_return_job_id == -1 or results[key][3] > results[max_mild_return_job_id][3]: 
                max_mild_return_job_id = key

        print("\n\n================ Overall Best Checkpoints ====================\n")
        print(f"Best success checkpoint: {self.checkpoint_list[max_success_job_id]} with {results[max_success_job_id][0]} successes")
        print(f"Best mild success checkpoint: {self.checkpoint_list[max_mild_success_job_id]} with {results[max_mild_success_job_id][1]} mild successes")
        print(f"Best return to box checkpoint: {self.checkpoint_list[max_return_job_id]} with {results[max_return_job_id][2]} returns to box")
        print(f"Best mild return to box checkpoint: {self.checkpoint_list[max_mild_return_job_id]} with {results[max_mild_return_job_id][3]} mild returns to box")
        print(f"Total Runs Evaluated Per Checkpoint: {len(self.seeds)}")
        print("\n=============================================================\n")