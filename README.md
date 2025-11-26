## Setup Instructions 
- Change the resolver to mamba for faster setup: `conda config --set solver libmamba` 
- Create a conda environment: `conda env create -f environment.yml` 

## Running the Basic Task
`python scripts/run_task_with_controller.py `
`python scripts/run_task_with_controller.py controller=ee_debug_controller.yaml`

Once you have saved data, you can visualize it using: `python scripts/run_task_with_controller.py run_methods="[check_saved_trajectory_images]"`
This will visualize the camera images stored using opencv rendering. 

Files in the `scripts/dev_scripts` directory do not currently follow hydra and can be used as scratch scripts to test code. 

For policy inference run `python scripts/run_task_with_controller.py run_methods="[run_eval]" controller=ee_diffusion_planar_controller` 