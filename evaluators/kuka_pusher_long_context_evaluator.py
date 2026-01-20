from evaluators.base_evaluator import BaseEvaluator
from tasks.kuka_pusher_long_context import KukaPlanarPusherLongContextBlock

from utils.file_utils import list_files_in_directory
from utils.logging_utils import IterTee
from utils.debug_utils import top_cuda_tensors

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import time
import sys
from multiprocessing import Process, Queue 
from queue import Empty 

from utils.diffusion_utils import load_policy

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

        self.debug = root_cfg.get("debug", False)

    def run_eval(self) -> None: 
        def eval_single_checkpoint(checkpoint_path: str, 
                                   output_dir: str,
                                   seeds: list[int],
                                   process_id: int, 
                                   out_q) -> None:
            import torch, gc
            torch.set_grad_enabled(False)

            import tracemalloc
            tracemalloc.start()
            snap = tracemalloc.take_snapshot()

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

            meta_meshcat = None 

            # load a policy once to get around memory leaks 
            sys.path.insert(0, self.root_cfg.controller.relative_path_to_diffusion_model)
            policy_and_cfg = load_policy(
                checkpoint_path, 
                load_normalizer_from_file=self.root_cfg.controller.load_normalizer_from_file, 
                infer_frozen_policy=self.root_cfg.controller.infer_frozen_policy
            )

            while m < len(seeds): 
                if m == 0: 
                    task: KukaPlanarPusherLongContextBlock = self.task_class(root_cfg=self.root_cfg)
                    meta_meshcat = task.meshcat
                else: 
                    task: KukaPlanarPusherLongContextBlock = self.task_class(root_cfg=self.root_cfg, 
                                                                             meshcat_initialized=meta_meshcat)

                controller = self.controller_class(root_cfg=self.root_cfg, 
                                                    policy_and_cfg=policy_and_cfg)

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

                with torch.inference_mode():
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

                meta_meshcat.Delete()
                controller.reset(None)
                controller.close()
                task.remove_controller()
                task.clear_attrs(skip_private=False, close_resources=True)
                del task 
                del controller 

                import gc, ctypes
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)

                def rss_mb():
                    with open("/proc/self/statm") as f:
                        rss_pages = int(f.read().split()[1])
                    return rss_pages * (os.sysconf("SC_PAGE_SIZE") / 1024**2)

                print("PID", os.getpid(), "RSS MB", rss_mb())

                snap2 = tracemalloc.take_snapshot()
                top = snap2.compare_to(snap, 'lineno')[:10]
                print("Top Python growth lines: ")
                for t in top: 
                    print(t)
                snap = snap2
                
                if self.debug: 
                    items = top_cuda_tensors(k=15)

                    if m % 3 == 0: 
                        breakpoint()

                    print(torch.cuda.memory_allocated() / 1e9, "GB allocated",
                        torch.cuda.memory_reserved() / 1e9, "GB reserved")

            
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

            out_q.put((process_id, (total_success, total_mild_success, total_return_to_box, total_mild_return_to_box)))

        # eval_single_checkpoint(
        #     self.checkpoint_list[0],
        #     self.checkpoint_output_dirs[0],
        #     self.seeds,
        #     0,
        #     Queue()
        # )

        out_q = Queue() 
        active = [] 
        proc_to_job = {}
        results = {}

        jobs = list(range(self.num_checkpoints))
        total = len(jobs)
        next_job = 0 

        while next_job < total or active: 
            while next_job < total and len(active) < self.num_processes: 
                job_id = next_job
                p = Process(
                    target = eval_single_checkpoint, 
                    args = (
                        self.checkpoint_list[next_job], 
                        self.checkpoint_output_dirs[next_job],
                        self.seeds,
                        job_id, 
                        out_q
                    )
                )
                p.start()
                active.append(p)
                proc_to_job[p] = job_id
                next_job += 1
            
            while True: 
                try: 
                    job_id, res = out_q.get_nowait()
                    results[job_id] = res 
                except Empty: 
                    break 
            
            still_active = []
            for p in active: 
                if p.is_alive(): 
                    still_active.append(p)
                else: 
                    p.join()
                    proc_to_job.pop(p, None)
            
            active = still_active

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