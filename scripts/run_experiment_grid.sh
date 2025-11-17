#!/bin/bash

# Define arrays for each parameter that varies
models=("vit")
normalizations=("layer" "none")
dropout_values=("0" "0.1")
seeds=("41" "42" "43" "11" "12")
learning_rates=("0.0001")  # Added learning rate grid

# Fixed parameters
dataset="tiny_imagenet"
tasks="40"
classes_per_task="5"
epochs_per_task="500"
wandb_tags="[main]"

# Number of GPUs to use
NUM_GPUS=4
BASE_GPU=1

# Calculate total number of experiments
total=$((${#models[@]} * ${#normalizations[@]} * ${#dropout_values[@]} * ${#seeds[@]} * ${#learning_rates[@]}))

echo "Starting $total individual experiment runs distributed across $NUM_GPUS GPUs..."
echo "--------------------------------------------------"

# Counter for experiments
job_count=0

# Loop through all combinations
for seed in "${seeds[@]}"; do
    for dropout in "${dropout_values[@]}"; do
        for model in "${models[@]}"; do
            for norm in "${normalizations[@]}"; do
                for lr in "${learning_rates[@]}"; do
                  # Determine which GPU to use (round-robin assignment)
                  gpu_id=$(( BASE_GPU + (job_count % NUM_GPUS) ))
                  
                  # Build the command
                  CMD="CUDA_VISIBLE_DEVICES=$gpu_id python scripts/run_experiment.py \
                    model=$model \
                    model.normalization=$norm \
                    model.dropout_p=$dropout \
                    dataset=$dataset \
                    training.tasks=$tasks \
                    training.classes_per_task=$classes_per_task \
                    training.epochs_per_task=$epochs_per_task \
                    training.seed=$seed \
                    optimizer.lr=$lr \
                    wandb_tags=$wandb_tags"
                  
                  echo "--------------------------------------------------"
                  echo "Launching (Job $((job_count + 1))/$total, Model: $model, Norm: $norm, Dropout: $dropout, Seed: $seed, LR: $lr, GPU: $gpu_id):"
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
    done
done

# Wait for any remaining background jobs to finish
echo "Waiting for all remaining background jobs to complete..."
wait

echo "--------------------------------------------------"
echo "All $total experiment runs launched and completed."
echo "--------------------------------------------------"