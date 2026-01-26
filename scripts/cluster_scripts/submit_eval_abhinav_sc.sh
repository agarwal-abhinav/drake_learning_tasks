#!/bin/bash 

echo "Starting evaluation script..."
source /etc/profile

# activate python environment
echo "Loading Anaconda module and activating environment..."
module load anaconda/Python-ML-2024b

echo "Activating conda environment..."
source activate drake-learning-tasks
export PYTHONNOUSERSITE=1

export HYDRA_FULL_ERROR=1

# add MOSEK for drake
export MOSEKLM_LICENSE_FILE=/home/gridsan/aagarwal2/mosek.lic

# Fix lack of X server when running on Supercloud
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export GALLIUM_DRIVER=llvmpipe
Xvfb "$DISPLAY" -screen 0 1400x900x24 -nolisten tcp > /tmp/xvfb.log 2>&1 &  # silence Xvfb output
xvfb_pid=$!
trap "kill $xvfb_pid" EXIT

python scripts/eval_multiple_checkpoints.py \
    hydra.run.dir=eval/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_48/mode_4/4_obs/ \
    evaluator.checkpoint_directory=/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_48/mode_4/4_obs/checkpoints/ \
    task.initial_location_type=4 evaluator.num_processes=5