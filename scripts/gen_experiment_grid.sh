#!/bin/bash

# Define arrays for each parameter that varies
models=("vit" "cnn" "resnet" "mlp")
normalizations=("batch" "layer" "none")
dropout_values=("0" "0.1")
seeds=("40" "41" "42" "43" "44")
learning_rates=("0.0001" "0.001" "0.01")  # Added learning rate grid

# Fixed parameters
dataset="tiny_imagenet"
tasks="40"
classes_per_task="5"
epochs_per_task="500"
wandb_tags="[main]"

# Counter for experiments
count=1
total=$((${#models[@]} * ${#normalizations[@]} * ${#dropout_values[@]} * ${#seeds[@]} * ${#learning_rates[@]}))

echo "Starting $total individual experiment runs..."

# Loop through all combinations
for model in "${models[@]}"; do
  for norm in "${normalizations[@]}"; do
    for dropout in "${dropout_values[@]}"; do
      for seed in "${seeds[@]}"; do
        for lr in "${learning_rates[@]}"; do  # Added learning rate loop
          # echo "Running experiment $count/$total: model=$model, normalization=$norm, dropout_p=$dropout, seed=$seed, lr=$lr"

          CMD="python scripts/run_experiment.py \
            model=$model \
            model.normalization=$norm \
            model.dropout_p=$dropout \
            dataset=$dataset \
            training.tasks=$tasks \
            training.classes_per_task=$classes_per_task \
            training.epochs_per_task=$epochs_per_task \
            training.seed=$seed \
            optimizer.lr=$lr \
            wandb_tags=$wandb_tags "
          
          echo $CMD
          echo ""

          count=$((count + 1))
        done
      done
    done
  done
done