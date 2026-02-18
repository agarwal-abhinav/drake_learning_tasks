#!/bin/bash

#SBATCH --job-name=de4_d_48_o_32_resnet
#SBATCH --time=23:59:00 
#SBATCH --cpus-per-task=30 
#SBATCH --mem=64G 
#SBATCH --output=submit_eval_vision.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Initialize and Load Modules
echo "[submit_eval_vision.sh] Loading modules and virtual environment"

echo "NODE: $SLURMD_NODENAME"
echo "JOB:  $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Exporting home directory, sourcing conda, and activating conda environment 
# currently home is set to where it should be and python is installed in scratch 
# porting this python to new home is the next goal 
export HOME=/data/locomotion/abhi_ag/
source /data/scratch-oc40/abhi_ag/python_environments/miniconda3/etc/profile.d/conda.sh

# activate the conda environment 
conda activate drake-learning-tasks
export PYTHONNOUSERSITE=1

# export somethings for help with multi-tasking
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1

# Export environment variables 
export HYDRA_FULL_ERROR=1

export MOSEKLM_LICENSE_FILE=/data/locomotion/abhi_ag/Licenses/mosek.lic

echo "[submit_eval_vision.sh] Running evaluation code..."

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_48/mode_4/80_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_48/mode_4/80_obs/
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_72/mode_4_0/32_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_72/mode_4_0/32_obs/4
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_48/mode_4_0/32_obs/checkpoints/
HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_48/mode_4_0/32_obs/4
RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0/47_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0/47_obs/4
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

python scripts/eval_multiple_checkpoints.py \
    hydra.run.dir=$HYDRA_RUN_DIR \
    evaluator.checkpoint_directory=$CHECKPOINT_DIR \
    task.initial_location_type=4 evaluator.num_processes=7 \
	controller.relative_path_to_diffusion_model=$RELATIVE_PATH_TO_DIFFUSION \
    controller.infer_frozen_policy=true
