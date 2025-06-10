#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
#SBATCH --output=output_%A_%a.log
#SBATCH --error=error_%A_%a.log
#SBATCH --mail-user=oliver.hausdoerfer@tum.de

#SBATCH --array=0-3

# Define the Python script
PYTHON_SCRIPT="main_optimize_scene.py"

# Define the arguments for each task
declare -a TASK0=('exp.sequence_index="565_81664_160332"' 'exp.experiment_folder="experiments_noSparse"' 'exp.l_optim_sparse_kp=0')
declare -a TASK1=('exp.sequence_index="565_81664_160332"' 'exp.experiment_folder="experiments_noInitShape"' 'exp.init_shape=false')
declare -a TASK2=('exp.sequence_index="565_81664_160332"' 'exp.experiment_folder="experiments_noInitShapeNoSparse"' 'exp.init_shape=false' 'exp.l_optim_sparse_kp=0')
declare -a TASK3=('exp.sequence_index="565_81664_160332"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_customDL"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# Determine which task to run based on array job ID
declare -n ARGS=TASK$SLURM_ARRAY_TASK_ID

# Execute the command with proper argument separation
srun --pty --container-mounts=./AnimalAvatar/:/home --container-image=/dss/dsshome1/0D/ge58guq2/olivertum+animal_avatar+0.2.sqsh python $PYTHON_SCRIPT ${ARGS[@]}


# Viable GPU nodes: lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
