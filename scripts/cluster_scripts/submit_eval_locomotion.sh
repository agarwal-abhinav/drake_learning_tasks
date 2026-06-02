#!/bin/bash

#SBATCH --job-name=fillm_eval
#SBATCH --time=40:00:00 
#SBATCH --cpus-per-task=30 
#SBATCH --mem=100G 
#SBATCH --output=submit_eval_locomotion.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=locomotion-h200 
#SBATCH --qos=locomotion-main
#SBATCH --gres=gpu:1

# Initialize and Load Modules
echo "[submit_eval_locomotion.sh] Loading modules and virtual environment"

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

echo "[submit_eval_locomotion.sh] Running evaluation code..."

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_72/mode_4/80_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_72/mode_4/80_obs/
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_48/mode_4_0_light_model/80_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_48/mode_4_0_light_model/80_obs/0
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0/48_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0/48_obs/0
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/constant_then_skip_second_frame_mode_4_0/40_obs/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/constant_then_skip_second_frame_mode_4_0/40_obs/0
# RELATIVE_PATH_TO_DIFFUSION=../gcs-diffusion/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/variable_context/two_modes/data_24/mode_4_0/variable_progressive_resnet_npast16/checkpoints/
# HYDRA_RUN_DIR=eval/iros/long_context_planar_pushing/variable_context/two_modes/data_24/mode_4_0/variable_progressive_resnet_npast16/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0/80_obs/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0/80_obs/checkpoints/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/limited_past/two_modes/data_24/mode_4_0/4_obs_no_past/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_24/mode_4_0/4_obs_no_past/checkpoints/0
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_48/mode_4_0/80_obs_no_past/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_48/mode_4_0/80_obs_no_past/checkpoints/0
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_24/mode_4_0/80_obs_full_attn/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_24/mode_4_0/80_obs_full_attn/checkpoints/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs_frozen4obs_resnet_init/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs_frozen4obs_resnet_init/checkpoints/0
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0/80_obs_frozen8obs_resnet_init/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0/80_obs_frozen8obs_resnet_init/checkpoints/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

# CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_48/mode_4_0/4_obs/checkpoints/
# HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_48/mode_4_0/4_obs/checkpoints/4
# RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/

CHECKPOINT_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_24/mode_4_0/80_obs_matched_params/checkpoints/
HYDRA_RUN_DIR=eval_2/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_24/mode_4_0/80_obs_matched_params/checkpoints/0
RELATIVE_PATH_TO_DIFFUSION=../diffusion-policy-experiment/


python scripts/eval_multiple_checkpoints.py \
    hydra.run.dir=$HYDRA_RUN_DIR \
    evaluator.checkpoint_directory=$CHECKPOINT_DIR \
    task.initial_location_type=0 evaluator.num_processes=7 \
	controller.relative_path_to_diffusion_model=$RELATIVE_PATH_TO_DIFFUSION \
    # controller.modes_to_eval="[0, 1, 3, 4]" controller.eval_max_time=70.0
