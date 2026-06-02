from evaluators.base_evaluator import BaseEvaluator
from tasks.kuka_pusher_long_context import KukaPlanarPusherLongContextBlock

from utils.file_utils import list_files_in_directory, return_highest_eval_seed_directory
from utils.logging_utils import IterTee
from utils.debug_utils import top_cuda_tensors

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import time
import sys
from multiprocessing import Process, Queue 
from queue import Empty 
from pathlib import Path
import yaml
import shutil

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

            snapshot_path = os.path.join(output_dir, "intermediate_snapshot.yaml")
            global_log_path = os.path.join(output_dir, "eval_log.txt")

            print(f"\n\nStarting evaluation for checkpoint: {checkpoint_path}\n using process id: {process_id}\n\n")

            if Path(output_dir).is_dir(): 
                # assert os.path.exists(snapshot_path), f"Output dir {output_dir} exists but no snapshot found."
                assert os.path.exists(global_log_path), f"Output dir {output_dir} exists but no global log found."

                if os.path.exists(snapshot_path): 
                    with open(snapshot_path, "r") as f:
                        intermediate_snapshot = yaml.safe_load(f)
                    
                    seeds_completed = intermediate_snapshot['seeds_completed']
                    if len(seeds_completed) == len(seeds): 
                        print(f"All seeds already completed for checkpoint {checkpoint_path}. Skipping evaluation.")
                        out_q.put((process_id, (intermediate_snapshot['total_success'], 
                                                intermediate_snapshot['total_mild_success'], 
                                                intermediate_snapshot['total_return_to_box'], 
                                                intermediate_snapshot['total_mild_return_to_box'])))
                        return
                    seeds_to_run = [s for s in seeds if s not in seeds_completed]
                    seeds_to_run = sorted(seeds_to_run)

                    global_log = open(global_log_path, "a")
                    global_log.write(f"\n\n=== Appending new eval run started: {time.asctime()} ===\n")
                    global_log.write(f"Resuming from seed: {seeds_to_run[0]}")
                    global_log.flush()
                    print(seeds_to_run)

                    total_success = intermediate_snapshot['total_success']
                    total_mild_success = intermediate_snapshot['total_mild_success']
                    total_return_to_box = intermediate_snapshot['total_return_to_box']
                    total_mild_return_to_box = intermediate_snapshot['total_mild_return_to_box']

                    # area metrics
                    total_mid_area = intermediate_snapshot['total_mid_area']
                    total_final_area = intermediate_snapshot['total_final_area']
                    num_mid_area = intermediate_snapshot['num_mid_area']
                    num_final_area = intermediate_snapshot['num_final_area']

                    highest_seed_started, highest_seed_dir_started = return_highest_eval_seed_directory(output_dir)

                    print(highest_seed_started)
                    if highest_seed_started == seeds_to_run[0]: 
                        print(f"Removing partially completed eval_seed_{highest_seed_started} directory at {highest_seed_dir_started}")
                        shutil.rmtree(highest_seed_dir_started)
                else: 
                    print("Removing partially completed output directory, job was terminated before snapshot could be saved. ")
                    shutil.rmtree(output_dir)

                    os.makedirs(output_dir, exist_ok=True)
                    seeds_to_run = seeds
                
                    global_log = open(global_log_path, "w")
                    global_log.write(f"=== Eval run started: {time.asctime()} ===\n")
                    global_log.flush()

                    total_success = 0
                    total_mild_success = 0 
                    total_return_to_box = 0 
                    total_mild_return_to_box = 0 

                    total_mid_area = 0 
                    total_final_area = 0 
                    num_mid_area = 0 
                    num_final_area = 0 

                    intermediate_snapshot = {
                        'seeds_completed': [], 
                        # metrics 
                        'total_success': 0, 
                        'total_mild_success': 0, 
                        'total_return_to_box': 0, 
                        'total_mild_return_to_box': 0, 
                        # area metrics
                        'total_mid_area': 0, 
                        'total_final_area': 0, 
                        'num_mid_area': 0, 
                        'num_final_area': 0
                    }

            else: 
                os.makedirs(output_dir, exist_ok=True)
                seeds_to_run = seeds
            
                global_log = open(global_log_path, "w")
                global_log.write(f"=== Eval run started: {time.asctime()} ===\n")
                global_log.flush()

                total_success = 0
                total_mild_success = 0 
                total_return_to_box = 0 
                total_mild_return_to_box = 0 

                total_mid_area = 0 
                total_final_area = 0 
                num_mid_area = 0 
                num_final_area = 0 

                intermediate_snapshot = {
                    'seeds_completed': [], 
                    # metrics 
                    'total_success': 0, 
                    'total_mild_success': 0, 
                    'total_return_to_box': 0, 
                    'total_mild_return_to_box': 0, 
                    # area metrics
                    'total_mid_area': 0, 
                    'total_final_area': 0, 
                    'num_mid_area': 0, 
                    'num_final_area': 0
                }
                
            tee = IterTee(sys.stdout, global_log)
            sys.stdout = tee 

            meta_meshcat = None 

            # load a policy once to get around memory leaks 
            sys.path.insert(0, self.root_cfg.controller.relative_path_to_diffusion_model)
            policy_and_cfg = load_policy(
                checkpoint_path, 
                load_normalizer_from_file=self.root_cfg.controller.load_normalizer_from_file, 
                infer_frozen_policy=self.root_cfg.controller.infer_frozen_policy
            )

            m = 0
            while m < len(seeds_to_run): 
                if m == 0: 
                    task: KukaPlanarPusherLongContextBlock = self.task_class(root_cfg=self.root_cfg)
                    meta_meshcat = task.meshcat
                else: 
                    task: KukaPlanarPusherLongContextBlock = self.task_class(root_cfg=self.root_cfg, 
                                                                             meshcat_initialized=meta_meshcat)

                controller = self.controller_class(root_cfg=self.root_cfg, 
                                                    policy_and_cfg=policy_and_cfg)

                task.controller = controller 

                os.mkdir(os.path.join(output_dir, f"eval_seed_{seeds_to_run[m]}"))
                iter_log_path = os.path.join(output_dir, f"eval_seed_{seeds_to_run[m]}", "eval_log.txt")
                iter_log = open(iter_log_path, "w")
                tee.set_iter_file(iter_log)

                if m < self.cfg.save_html_first: 
                    save_html = True 
                else: 
                    save_html = False 
                
                print(f"\n--- Starting eval for seed {seeds_to_run[m]}---\n")
                task.reset_robot(seeds_to_run[m])

                with torch.inference_mode():
                    areas, successes, correct_returns = task.diffusion_rollout(
                        self.cfg.eval_max_time, 
                        save_path = os.path.join(output_dir, f"eval_seed_{seeds_to_run[m]}"), 
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

                intermediate_snapshot['seeds_completed'].append(seeds_to_run[m])
                intermediate_snapshot['total_success'] = total_success
                intermediate_snapshot['total_mild_success'] = total_mild_success
                intermediate_snapshot['total_return_to_box'] = total_return_to_box
                intermediate_snapshot['total_mild_return_to_box'] = total_mild_return_to_box

                # area metrics 
                intermediate_snapshot['total_mid_area'] = total_mid_area
                intermediate_snapshot['total_final_area'] = total_final_area
                intermediate_snapshot['num_mid_area'] = num_mid_area
                intermediate_snapshot['num_final_area'] = num_final_area

                with open(snapshot_path, "w") as f:
                    yaml.dump(intermediate_snapshot, f)

                m+=1 

                meta_meshcat.Delete()
                controller.reset(None)
                controller.close()
                task.remove_controller()
                task.clear_attrs(skip_private=False, close_resources=True)
                del task 
                del controller 
                
                # some system level debugging information
                import gc, ctypes
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)

                def rss_mb():
                    with open("/proc/self/statm") as f:
                        rss_pages = int(f.read().split()[1])
                    return rss_pages * (os.sysconf("SC_PAGE_SIZE") / 1024**2)

                print("PID", os.getpid(), "RSS MB", rss_mb())
                try:
                    aff = os.sched_getaffinity(0)
                    aff_n = len(aff)
                except Exception:
                    aff_n = None

                print(
                    f"[job {job_id}] pid={os.getpid()} "
                    f"rss={rss_mb():.1f}MB "
                    f"cpu_affinity={aff_n}"
                )

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