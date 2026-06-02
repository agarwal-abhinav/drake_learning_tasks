#!/bin/bash

#SBATCH --job-name=dit_re
#SBATCH --time=23:59:00 
#SBATCH --cpus-per-task=30 
#SBATCH --mem=90G 
#SBATCH --output=submit_eval_vision.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=vision-shared-a100
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
source /data/locomotion/abhi_ag/miniconda3/etc/profile.d/conda.sh

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

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_96/mode_4_0/4_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_96/mode_4_0/4_obs/0
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs_no_past_frozen_resnet/checkpoints
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs_no_past_frozen_resnet/checkpoints/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/limited_past/two_modes/data_48/mode_4_0/4_obs_no_past/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_48/mode_4_0/4_obs_no_past/checkpoints/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs_frozen_resnet_init_from_8obs/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs_frozen_resnet_init_from_8obs_ptp_realign/checkpoints/0
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_24/mode_4_0/80_obs_full_attn/checkpoints/
HYDRA_RUN_DIR=eval_4/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_24/mode_4_0/80_obs_full_attn/checkpoints/0
RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# python scripts/eval_multiple_checkpoints.py \
#     hydra.run.dir=$HYDRA_RUN_DIR \
#     evaluator.checkpoint_directory=$CHECKPOINT_DIR \
#     task.initial_location_type=0 evaluator.num_processes=7 \
# 	controller.relative_path_to_diffusion_model=$RELATIVE_PATH_TO_DIFFUSION \
#     controller.infer_frozen_policy=true

python scripts/eval_multiple_checkpoints.py \
    hydra.run.dir=$HYDRA_RUN_DIR \
    evaluator.checkpoint_directory=$CHECKPOINT_DIR \
    task.initial_location_type=0 evaluator.num_processes=7 \
	controller.relative_path_to_diffusion_model=$RELATIVE_PATH_TO_DIFFUSION 
    # controller.modes_to_eval="[0, 1, 3, 4]" controller.eval_max_time=70.0