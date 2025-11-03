#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
#SBATCH --output=output_%A_%a.log
#SBATCH --error=error_%A_%a.log
#SBATCH --mail-user=oliver.hausdoerfer@tum.de

#SBATCH --array=0

# Define the Python script
PYTHON_SCRIPT="main_optimize_scene.py"

# Define the arguments for each task
# NOTE it is save to save every experiment in a different exp.experiment_folder. Otherwise runs might interfere with their save folders.

# declare -a TASK0=('exp.sequence_index="left_right_turn_2058226999"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila_left_right_turn_2058226999"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

declare -a TASK0=('exp.sequence_index="slow_1313807000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila_slow_1313807000"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# declare -a TASK1=('exp.sequence_index="slow_turn_1771233000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila_slow_turn_1771233000"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# declare -a TASK3=('exp.sequence_index="start_stop_785558000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila_d938"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# declare -a TASK2=('exp.sequence_index="start_stop_1271493000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila_start_stop_1271493000"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# declare -a TASK5=('exp.sequence_index="turn_left_1771233000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# declare -a TASK6=('exp.sequence_index="turn_right_1771233000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# declare -a TASK7=('exp.sequence_index="walk_869488000"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_maila"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')


# Determine which task to run based on array job ID
declare -n ARGS=TASK$SLURM_ARRAY_TASK_ID

# Execute the command with proper argument separation
srun --pty --container-mounts=./AnimalAvatar/:/home --container-image=/dss/dsshome1/0D/ge58guq2/olivertum+animal_avatar+0.2.sqsh python $PYTHON_SCRIPT ${ARGS[@]}


# Viable GPU nodes: lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
