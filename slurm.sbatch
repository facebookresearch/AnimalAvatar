#!/bin/bash

# Define the Python script
PYTHON_SCRIPT="main_optimize_scene.py"

# Define the arguments for each task
declare -a TASK0=('exp.sequence_index="565_81664_160332"' 'exp.experiment_folder="experiments_noSparse"' 'exp.l_optim_sparse_kp=0')
declare -a TASK1=('exp.sequence_index="565_81664_160332"' 'exp.experiment_folder="experiments_noInitShape"' 'exp.init_shape=false')
declare -a TASK2=('exp.sequence_index="565_81664_160332"' 'exp.experiment_folder="experiments_noInitShapeNoSparse"' 'exp.init_shape=false' 'exp.l_optim_sparse_kp=0')
declare -a TASK3=('exp.sequence_index="565_81664_160332"' 'exp.dataset_source="CUSTOM"' 'exp.experiment_folder="experiments_customDL"' 'exp.l_optim_sparse_kp=0' 'exp.init_shape=false')

# Test each command format
echo "Testing command formats:"
echo "------------------------"

for i in {0..3}; do
  # Use nameref to select the appropriate task array
  declare -n ARGS=TASK$i
  
  # Uncomment this line to actually execute the commands (for testing only)
  echo python $PYTHON_SCRIPT ${ARGS[@]}

  echo \n
  echo \n
done