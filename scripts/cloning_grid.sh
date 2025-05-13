#!/bin/bash

# Run from the project root directory
set -e # Exit on first error

BASE_PYTHON_CMD="python scripts/run_experiment.py"
BASE_CONFIG="model=mlp training=cloning training.initial_epochs=10 training.epochs_per_expansion=20 model.dropout_p=0 optimizer=noisysgd"

# Define parameter grids
NOISE_SCALES=("0.01" "0.02")
NOISE_DECAYS=("0" "0.9" "0.99" "0.999" "1.0")
NUM_GPUS=4
BASE_GPU=1

# Define seed range
START_SEED=0
END_SEED=4  # Seeds 0 to 4 (5 seeds total)

# Calculate total runs
TOTAL_RUNS=$((${#NOISE_SCALES[@]} * ${#NOISE_DECAYS[@]} * (END_SEED - START_SEED + 1)))
echo "Starting experiment grid: run_experiment.py"
echo "Seeds: $(($END_SEED - $START_SEED + 1)) (from $START_SEED to $END_SEED)"
echo "Noise scales: ${#NOISE_SCALES[@]}"
echo "Noise decays: ${#NOISE_DECAYS[@]}"
echo "Total runs to launch: $TOTAL_RUNS"
echo "Distributing across $NUM_GPUS GPUs."
echo "--------------------------------------------------"

job_count=0

for seed_val in $(seq $START_SEED $END_SEED)
do
    SEED_OVERRIDE="training.seed=$seed_val"
    
    for noise_scale in "${NOISE_SCALES[@]}"
    do
        for noise_decay in "${NOISE_DECAYS[@]}"
        do
            # Assign GPU: cuda:0, cuda:1, cuda:2, cuda:3 in a round-robin fashion
            gpu_id=$(($job_count % $NUM_GPUS + $BASE_GPU))
            DEVICE_OVERRIDE="training.device=cuda:$gpu_id"
        
            # Build the command
            CMD="$BASE_PYTHON_CMD $BASE_CONFIG $DEVICE_OVERRIDE $SEED_OVERRIDE optimizer.noise_scale=$noise_scale optimizer.noise_decay=$noise_decay wandb_project=cloning wandb_tags=[main]"
        
            # Add a descriptive run name for wandb if it's being used
            WANDB_RUN_NAME="cloning_seed${seed_val}_ns${noise_scale}_nd${noise_decay}_gpu${gpu_id}"
            CMD="$CMD logging.wandb_run_name=$WANDB_RUN_NAME"
        
            echo "--------------------------------------------------"
            echo "Launching (Job $((job_count + 1))/$TOTAL_RUNS, Seed: $seed_val, NS: $noise_scale, ND: $noise_decay, GPU: cuda:$gpu_id):"
            echo "CMD: $CMD"
            echo "--------------------------------------------------"
        
            # Run in background
            eval $CMD &
        
            job_count=$((job_count + 1))
        
            # If we've launched NUM_GPUS jobs, wait for them to complete before launching more
            if [ $(($job_count % $NUM_GPUS)) -eq 0 ] && [ $job_count -ne 0 ]; then
                echo "Launched $NUM_GPUS jobs, waiting for this batch to complete..."
                wait
                echo "Batch completed. Proceeding with next batch."
            fi
        done
    done
done

# Wait for any remaining background jobs to finish
echo "Waiting for all remaining background jobs to complete..."
wait

echo "--------------------------------------------------"
echo "All $TOTAL_RUNS experiment runs launched and completed."
echo "--------------------------------------------------"