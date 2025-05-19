#!/bin/bash

# Define arrays for each parameter that varies
models=("mlp" "cnn" "vit" "resnet")
normalizations=("batch" "layer" "none")
dropout_values=("0" "0.1" "0.2")
seeds=("40" "41" "42" "43" "44")
optimizers=("sgd" "adam")
learning_rates=("0.01" "0.001" "0.0001")  # Added learning rate grid

# Fixed parameters
num_workers="2"
dataset="cifar10"
initial_epochs="20"
epochs_per_expansion="500"
wandb_tags="[main,cloning]"

# Counter for experiments
count=1
total=$((${#models[@]} * ${#normalizations[@]} * ${#dropout_values[@]} * ${#seeds[@]} * ${#learning_rates[@]}))

echo "Starting $total individual experiment runs..."

# Loop through all combinations
for model in "${models[@]}"; do
  for norm in "${normalizations[@]}"; do
    for dropout in "${dropout_values[@]}"; do
      for seed in "${seeds[@]}"; do
	for optimizer in "${optimizers[@]}"; do
        for lr in "${learning_rates[@]}"; do  # Added learning rate loop
          # echo "Running experiment $count/$total: model=$model, normalization=$norm, dropout_p=$dropout, seed=$seed, lr=$lr"

	  CMD="$(which python) $(pwd)/scripts/run_experiment.py \
            model=$model \
            model.normalization=$norm \
            model.dropout_p=$dropout \
            dataset=$dataset \
	    training=cloning \
	    training.initial_epochs=$initial_epochs \
            training.epochs_per_expansion=$epochs_per_expansion \
	    training.num_workers=$num_workers \ 
            training.seed=$seed \
	    optimizer=$optimizer \
            optimizer.lr=$lr \
            wandb_tags=$wandb_tags "
          
          echo gpujob submit \"$CMD\"
	  gpujob submit "$CMD"

          echo ""

          count=$((count + 1))
        done
        done
      done
    done
  done
done
