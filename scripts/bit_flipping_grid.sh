#!/bin/bash

# Run from the project root directory.
# Example: bash scripts/bit_flipping_grid.sh

set -e # Exit on first error

BASE_PYTHON_CMD="python scripts/run_bit_flipping_experiment.py"
N_EXAMPLES_OVERRIDE="training.n_examples=5000000" # 5e6

START_SEED=20
END_SEED=99 # Seeds 0 to 99

NUM_GPUS=4

declare -a TRAINING_CONFIG_SETS
# Config 1: Default
#TRAINING_CONFIG_SETS+=( "default" "" )
# Config 2: continual_backprop=true
TRAINING_CONFIG_SETS+=( "cbp" "training.continual_backprop=true" )
# Config 3: use_shrink_perturb=true
#TRAINING_CONFIG_SETS+=( "sp" "training.use_shrink_perturb=true" )
NUM_CONFIG_SETS=1

TOTAL_RUNS=$(($END_SEED - $START_SEED + 1))

echo "Starting experiment grid: run_bit_flipping_experiment.py"
echo "Seeds per config: $(($END_SEED - $START_SEED + 1))"
echo "Training configurations: $NUM_CONFIG_SETS"
echo "Total runs to launch: $TOTAL_RUNS"
echo "Distributing across $NUM_GPUS GPUs."
echo "--------------------------------------------------"

job_count=0

for seed_val in $(seq $START_SEED $END_SEED)
do
    CURRENT_SEED_OVERRIDE="training.seed=$seed_val"

    config_name_suffix=${TRAINING_CONFIG_SETS[0]}
    specific_training_overrides=${TRAINING_CONFIG_SETS[1]}

    # Assign GPU: cuda:0, cuda:1, cuda:2, cuda:3 in a round-robin fashion
    gpu_id=$(($job_count % $NUM_GPUS))
    DEVICE_OVERRIDE="training.device=cuda:$gpu_id"

    WANDB_RUN_NAME_OVERRIDE="logging.wandb_run_name=bf_seed${seed_val}_${config_name_suffix}_n5e6_gpu${gpu_id}"

    CMD="$BASE_PYTHON_CMD $N_EXAMPLES_OVERRIDE $CURRENT_SEED_OVERRIDE $DEVICE_OVERRIDE $WANDB_RUN_NAME_OVERRIDE wandb_tags=[mainv3]"

    if [ -n "$specific_training_overrides" ]; then
        CMD="$CMD $specific_training_overrides"
    fi
    
    echo "--------------------------------------------------"
    echo "Launching (Job $((job_count + 1))/$TOTAL_RUNS, Seed: $seed_val, Config: $config_name_suffix, GPU: cuda:$gpu_id):"
    echo "CMD: $CMD"
    echo "--------------------------------------------------"
    
    # Run in background
    eval $CMD & 
    
    job_count=$((job_count + 1))

    # If we've launched NUM_GPUS jobs, wait for them to complete before launching more
    # More sophisticated: wait for *any* job to finish if job_count >= NUM_GPUS
    if [ $(($job_count % $NUM_GPUS)) -eq 0 ] && [ $job_count -ne 0 ]; then
        echo "Launched $NUM_GPUS jobs, waiting for this batch to complete..."
        wait
        echo "Batch completed. Proceeding with next batch."
    fi

done

# Wait for any remaining background jobs to finish
echo "Waiting for all remaining background jobs to complete..."
wait

echo "--------------------------------------------------"
echo "All $TOTAL_RUNS experiment runs launched and completed."
echo "--------------------------------------------------"
