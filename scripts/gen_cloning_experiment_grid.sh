#!/bin/bash

# Define arrays for each parameter that varies
models=("mlp")
normalizations=("batch" "layer" "none")
dropout_values=("0")
seeds=("41" "42" "43")
optimizers=("noisysgd")
noise_scales=("0" "0.01", "0.02")
noise_decays=("0" "0.9" "0.99" "0.999")
learning_rates=("0.01")  # Added learning rate grid

# Fixed parameters
dataset="cifar10"
initial_epochs="10"
epochs_per_expansion="100"
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
	for noise_scale in "${noise_scales[@]}"; do
	for noise_decay in "${noise_decays[@]}"; do
          # echo "Running experiment $count/$total: model=$model, normalization=$norm, dropout_p=$dropout, seed=$seed, lr=$lr"

	  CMD="$(which python) $(pwd)/scripts/run_experiment.py \
            model=$model \
            model.normalization=$norm \
            model.dropout_p=$dropout \
            dataset=$dataset \
	    training=cloning \
	    training.initial_epochs=$initial_epochs \
            training.epochs_per_expansion=$epochs_per_expansion \
            training.seed=$seed \
	    optimizer=$optimizer \
	    optimizer.noise_decay=$noise_decay \
	    optimizer.noise_scale=$noise_scale \
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
  done
done
