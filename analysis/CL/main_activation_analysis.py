"""
Continual learning experiment focusing on neural network activation analysis.
This script demonstrates how to track and analyze activations during continual learning.
"""

import os
import torch
import torch.nn as nn
from datetime import datetime

from utils.config import ExperimentConfig, create_model_from_config
from data.datasets import ContinualTaskSequence, ContinualDataLoader
from training.trainer import ContinualTrainer
from training.metrics import get_layer_activations
from analysis.activation_analysis import (
    analyze_activations, 
    visualize_activation_distributions,
    visualize_feature_space,
    visualize_feature_correlations,
    compare_activations
)

def main():
    # Generate timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name=f"activation_analysis_{timestamp}",
        random_seed=42,
        dataset_name="mnist",  # Using MNIST for clear feature visualization
        task_type="pairs",
        num_tasks=5,
        model_type="mlp",
        model_config={
            "hidden_sizes": [256, 128, 64],
            "activation": "relu",
            "dropout_p": 0.0,  # No dropout for clearer activation analysis
            "normalization": None  # No normalization for clearer activation patterns
        },
        optimizer_type="adam",
        optimizer_config={"lr": 0.001},
        batch_size=128,
        num_epochs_per_task=5,
        metrics_config={
            "record_activations": True,
            "record_activations_freq": 1,  # Record every epoch
        },
        results_dir=f"./results/activation_analysis_{timestamp}",
        checkpoint_dir=f"./checkpoints/activation_analysis_{timestamp}"
    )
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Create continual learning task sequence
    task_sequence = ContinualTaskSequence(
        dataset_name=config.dataset_name,
        task_type=config.task_type,
        n_tasks=config.num_tasks,
        root=config.dataset_root
    )
    
    # Create data loader
    data_loader = ContinualDataLoader(
        task_sequence=task_sequence,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    model = create_model_from_config(config)
    model.to(device)
    
    # Define optimizer factory
    def optimizer_factory(model):
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer_config["lr"],
            weight_decay=config.optimizer_config.get("weight_decay", 0)
        )
    
    # Create trainer
    trainer = ContinualTrainer(
        model=model,
        continual_data_loader=data_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer_factory=optimizer_factory,
        device=device,
        metrics_config=config.metrics_config,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # Store initial activations before training
    print("Recording initial activations before training...")
    # Get a batch from the first task
    first_task_loader = data_loader.get_task_loader(0, train=False)
    sample_batch = next(iter(first_task_loader))
    sample_inputs, sample_labels = sample_batch[0].to(device), sample_batch[1].to(device)
    
    # Get initial activations
    with torch.no_grad():
        initial_outputs, initial_activations = model(sample_inputs, store_activations=True)
    
    # Train on all tasks
    print(f"Starting training on {config.num_tasks} tasks...")
    history = trainer.train_all_tasks(num_epochs_per_task=config.num_epochs_per_task)
    
    # Collect and analyze final activations using the same batch
    print("Collecting and analyzing final activations...")
    with torch.no_grad():
        final_outputs, final_activations = model(sample_inputs, store_activations=True)
    
    # Analyze activations from initial and final states
    initial_stats = analyze_activations(initial_activations)
    final_stats = analyze_activations(final_activations)
    
    # Print some key activation statistics
    print("\nActivation Statistics Comparison (Initial vs. Final):")
    
    # Find common layers to compare
    common_layers = set(initial_stats.keys()).intersection(set(final_stats.keys()))
    for layer in common_layers:
        # Skip input layer
        if layer == 'input':
            continue
            
        print(f"\nLayer: {layer}")
        initial_mean = initial_stats[layer].get('mean', 'N/A')
        final_mean = final_stats[layer].get('mean', 'N/A')
        print(f"  Mean activation: {initial_mean:.4f} → {final_mean:.4f}")
        
        initial_std = initial_stats[layer].get('std', 'N/A')
        final_std = final_stats[layer].get('std', 'N/A')
        print(f"  Std deviation: {initial_std:.4f} → {final_std:.4f}")
        
        if 'dead_fraction' in initial_stats[layer] and 'dead_fraction' in final_stats[layer]:
            initial_dead = initial_stats[layer]['dead_fraction']
            final_dead = final_stats[layer]['dead_fraction']
            print(f"  Dead neurons: {initial_dead:.2%} → {final_dead:.2%}")
        
        if 'feature_entropy' in initial_stats[layer] and 'feature_entropy' in final_stats[layer]:
            initial_entropy = initial_stats[layer]['feature_entropy']
            final_entropy = final_stats[layer]['feature_entropy']
            print(f"  Feature entropy: {initial_entropy:.4f} → {final_entropy:.4f}")
    
    # Visualize activation distributions for each layer
    print("\nGenerating activation distribution visualizations...")
    for layer in common_layers:
        if layer != 'input' and layer != 'output':
            visualize_activation_distributions(
                {layer: final_activations[layer]},
                figsize=(12, 6),
                save_path=os.path.join(config.results_dir, f"{layer}_activation_distribution.png")
            )
    
    # Visualize feature space (t-SNE) for a selected hidden layer
    print("\nGenerating feature space visualization...")
    # Choose a good intermediate layer
    feature_layer = None
    for layer in final_activations:
        if 'fc_' in layer or 'linear_' in layer:
            if layer != 'output' and layer != 'input':
                feature_layer = layer
                break
    
    if feature_layer:
        visualize_feature_space(
            final_activations, 
            feature_layer,
            n_components=2, 
            method='tsne', 
            targets=sample_labels.cpu(),
            save_path=os.path.join(config.results_dir, f"{feature_layer}_tsne.png")
        )
    
    # Compare activations between initial and final state
    print("\nComparing initial and final activations...")
    # Choose a good intermediate layer for comparison
    for layer in common_layers:
        if layer != 'input' and layer != 'output':
            _, similarities = compare_activations(
                {layer: initial_activations[layer]}, 
                {layer: final_activations[layer]},
                metric='cosine',
                save_path=os.path.join(config.results_dir, f"{layer}_activation_comparison.png")
            )
    
    # For one of the layers, visualize feature correlations
    print("\nGenerating feature correlation heatmap...")
    if feature_layer:
        visualize_feature_correlations(
            final_activations,
            feature_layer,
            max_features=50,  # Limit number of features to avoid cluttered visualization
            save_path=os.path.join(config.results_dir, f"{feature_layer}_correlation_heatmap.png")
        )
    
    # Analyze activations across different tasks
    print("\nAnalyzing activations across different tasks...")
    task_activations = {}
    
    # Get activations for each task
    for task_idx in range(config.num_tasks):
        task_loader = data_loader.get_task_loader(task_idx, train=False)
        batch = next(iter(task_loader))
        task_inputs = batch[0].to(device)
        task_labels = batch[1]
        
        with torch.no_grad():
            _, task_activation = model(task_inputs, store_activations=True)
        
        task_activations[f"task_{task_idx}"] = task_activation
    
    # Compare activations between tasks
    for i in range(config.num_tasks - 1):
        for j in range(i + 1, config.num_tasks):
            task_i = f"task_{i}"
            task_j = f"task_{j}"
            
            if feature_layer:
                # Compare the feature layer activations between tasks
                compare_activations(
                    {feature_layer: task_activations[task_i][feature_layer]},
                    {feature_layer: task_activations[task_j][feature_layer]},
                    figsize=(10, 6),
                    save_path=os.path.join(config.results_dir, f"{feature_layer}_{task_i}_vs_{task_j}.png")
                )
    
    print(f"\nExperiment complete! Results saved to {config.results_dir}")

if __name__ == "__main__":
    main()