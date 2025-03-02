"""
Continual learning experiment comparing different model architectures.
This script demonstrates how to compare MLP, CNN, and ResNet on the same continual learning tasks.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.config import ExperimentConfig, create_model_from_config
from data.datasets import ContinualTaskSequence, ContinualDataLoader
from training.trainer import ContinualTrainer
import utils.visualization as viz

def run_model_experiment(model_type, model_config, data_loader, base_config, result_dir):
    """Run experiment for a specific model architecture."""
    print(f"\n{'='*50}")
    print(f"Running experiment with {model_type} model")
    print(f"{'='*50}")
    
    # Update config for this model
    config = ExperimentConfig(
        experiment_name=f"{model_type}_{base_config.dataset_name}_{base_config.task_type}",
        random_seed=base_config.random_seed,
        dataset_name=base_config.dataset_name,
        task_type=base_config.task_type,
        num_tasks=base_config.num_tasks,
        model_type=model_type,
        model_config=model_config,
        optimizer_type=base_config.optimizer_type,
        optimizer_config=base_config.optimizer_config,
        batch_size=base_config.batch_size,
        num_epochs_per_task=base_config.num_epochs_per_task,
        results_dir=result_dir,
        checkpoint_dir=os.path.join(base_config.checkpoint_dir, model_type)
    )
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    model = create_model_from_config(config)
    model.to(device)
    
    # Define optimizer factory
    def optimizer_factory(model):
        if config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.optimizer_config["lr"],
                weight_decay=config.optimizer_config.get("weight_decay", 0)
            )
        else:
            return torch.optim.SGD(
                model.parameters(),
                lr=config.optimizer_config["lr"],
                momentum=config.optimizer_config.get("momentum", 0.9),
                weight_decay=config.optimizer_config.get("weight_decay", 0)
            )
    
    # Create trainer
    trainer = ContinualTrainer(
        model=model,
        continual_data_loader=data_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer_factory=optimizer_factory,
        device=device,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # Train on all tasks
    print(f"Starting training {model_type} on {config.num_tasks} tasks...")
    history = trainer.train_all_tasks(num_epochs_per_task=config.num_epochs_per_task)
    
    # Final evaluation on all tasks
    print(f"\nFinal evaluation of {model_type} on all tasks:")
    final_accuracies = trainer.evaluate_all_tasks()
    
    # Compute forgetting
    forgetting = trainer.compute_forgetting()
    print(f"\n{model_type} forgetting metrics:")
    for task, value in forgetting.items():
        if task != "average":
            print(f"{task}: {value:.4f}")
    print(f"Average forgetting: {forgetting.get('average', 0):.4f}")
    
    return {
        'model_type': model_type,
        'history': history,
        'final_accuracies': final_accuracies,
        'forgetting': forgetting
    }

def main():
    # Generate timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base configuration
    base_config = ExperimentConfig(
        experiment_name=f"model_comparison_{timestamp}",
        random_seed=42,
        dataset_name="cifar10",
        task_type="pairs",
        num_tasks=5,
        optimizer_type="adam",
        optimizer_config={"lr": 0.001, "weight_decay": 1e-4},
        batch_size=128,
        num_epochs_per_task=5,
        results_dir=f"./results/model_comparison_{timestamp}",
        checkpoint_dir=f"./checkpoints/model_comparison_{timestamp}"
    )
    
    # Create directories
    os.makedirs(base_config.results_dir, exist_ok=True)
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)
    
    # Create task sequence (same for all models)
    task_sequence = ContinualTaskSequence(
        dataset_name=base_config.dataset_name,
        task_type=base_config.task_type,
        n_tasks=base_config.num_tasks,
        root=base_config.dataset_root
    )
    
    # Create data loader (same for all models)
    data_loader = ContinualDataLoader(
        task_sequence=task_sequence,
        batch_size=base_config.batch_size,
        num_workers=base_config.num_workers
    )
    
    # Model configurations to test
    model_configs = {
        "mlp": {
            "hidden_sizes": [512, 256, 128],
            "activation": "relu",
            "dropout_p": 0.2,
            "normalization": "batch"
        },
        "cnn": {
            "conv_channels": [64, 128, 256],
            "kernel_sizes": [3, 3, 3],
            "fc_hidden_units": [512],
            "activation": "relu",
            "dropout_p": 0.2,
            "use_batchnorm": True
        },
        "resnet": {
            "layers": [2, 2, 2, 2],  # ResNet-18
            "activation": "relu",
            "dropout_p": 0.1,
            "use_batchnorm": True
        }
    }
    
    # Run experiments for each model type
    results = {}
    for model_type, model_config in model_configs.items():
        result = run_model_experiment(
            model_type=model_type,
            model_config=model_config,
            data_loader=data_loader,
            base_config=base_config,
            result_dir=base_config.results_dir
        )
        results[model_type] = result
    
    # Compare performance across models
    print("\nComparing model performances:")
    avg_accuracies = {model: results[model]['final_accuracies'].get('average', 0) 
                     for model in results}
    avg_forgetting = {model: results[model]['forgetting'].get('average', 0) 
                     for model in results}
    
    for model in avg_accuracies:
        print(f"{model}: Accuracy = {avg_accuracies[model]:.4f}, Forgetting = {avg_forgetting[model]:.4f}")
    
    # Visualize comparisons
    print("\nGenerating comparison visualizations...")
    
    # Final task accuracies comparison
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    accuracies = [avg_accuracies[model] for model in model_names]
    plt.bar(model_names, accuracies)
    plt.ylim(0, 1.0)
    plt.title('Average Accuracy Across Tasks by Model Architecture')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', alpha=0.3)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.savefig(os.path.join(base_config.results_dir, "model_accuracy_comparison.png"))
    
    # Forgetting comparison
    plt.figure(figsize=(12, 6))
    forgetting_values = [avg_forgetting[model] for model in model_names]
    plt.bar(model_names, forgetting_values)
    plt.title('Average Forgetting by Model Architecture')
    plt.ylabel('Forgetting')
    plt.grid(axis='y', alpha=0.3)
    for i, f in enumerate(forgetting_values):
        plt.text(i, f + 0.01, f'{f:.4f}', ha='center')
    plt.savefig(os.path.join(base_config.results_dir, "model_forgetting_comparison.png"))
    
    # Save individual model performance plots
    for model_type in results:
        viz.plot_task_performance(
            results[model_type]['final_accuracies'],
            save_path=os.path.join(base_config.results_dir, f"{model_type}_task_performance.png")
        )
        
        viz.plot_forgetting_curves(
            results[model_type]['history'],
            save_path=os.path.join(base_config.results_dir, f"{model_type}_forgetting_curves.png")
        )
    
    print(f"\nExperiment complete! Results saved to {base_config.results_dir}")

if __name__ == "__main__":
    main()