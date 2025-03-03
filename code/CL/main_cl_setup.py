"""
Compare different continual learning setups (sequential vs. pairs).
This script demonstrates the impact of different task arrangements on continual learning.
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

def run_cl_experiment(task_type, data_loader, base_config, result_dir):
    """Run experiment for a specific continual learning setup."""
    print(f"\n{'='*50}")
    print(f"Running experiment with {task_type} task arrangement")
    print(f"{'='*50}")
    
    # Update config for this experiment
    config = ExperimentConfig(
        experiment_name=f"{base_config.model_type}_{base_config.dataset_name}_{task_type}",
        random_seed=base_config.random_seed,
        dataset_name=base_config.dataset_name,
        task_type=task_type,
        num_tasks=base_config.num_tasks,
        model_type=base_config.model_type,
        model_config=base_config.model_config,
        optimizer_type=base_config.optimizer_type,
        optimizer_config=base_config.optimizer_config,
        batch_size=base_config.batch_size,
        num_epochs_per_task=base_config.num_epochs_per_task,
        results_dir=result_dir,
        checkpoint_dir=os.path.join(base_config.checkpoint_dir, task_type)
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
    print(f"Starting training on {config.num_tasks} tasks with {task_type} arrangement...")
    history = trainer.train_all_tasks(num_epochs_per_task=config.num_epochs_per_task)
    
    # Final evaluation on all tasks
    print(f"\nFinal evaluation ({task_type} arrangement) on all tasks:")
    final_accuracies = trainer.evaluate_all_tasks()
    
    # Compute forgetting
    forgetting = trainer.compute_forgetting()
    print(f"\n{task_type} forgetting metrics:")
    for task, value in forgetting.items():
        if task != "average":
            print(f"\n{task_type} forgetting metrics:")
    for task, value in forgetting.items():
        if task != "average":
            print(f"{task}: {value:.4f}")
    print(f"Average forgetting: {forgetting.get('average', 0):.4f}")
    
    return {
        'task_type': task_type,
        'history': history,
        'final_accuracies': final_accuracies,
        'forgetting': forgetting
    }

def main():
    # Generate timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base configuration
    base_config = ExperimentConfig(
        experiment_name=f"cl_setup_comparison_{timestamp}",
        random_seed=42,
        dataset_name="cifar10",
        model_type="cnn",  # CNN works well for this comparison
        model_config={
            "conv_channels": [64, 128, 256],
            "kernel_sizes": [3, 3, 3],
            "fc_hidden_units": [512],
            "activation": "relu",
            "dropout_p": 0.2,
            "use_batchnorm": True
        },
        num_tasks=5,
        optimizer_type="adam",
        optimizer_config={"lr": 0.001, "weight_decay": 1e-4},
        batch_size=128,
        num_epochs_per_task=5,
        results_dir=f"./results/cl_setup_comparison_{timestamp}",
        checkpoint_dir=f"./checkpoints/cl_setup_comparison_{timestamp}"
    )
    
    # Create directories
    os.makedirs(base_config.results_dir, exist_ok=True)
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)
    
    # Task arrangements to compare
    task_arrangements = ["pairs", "sequential"]
    
    # Run experiments for each task arrangement
    results = {}
    for task_type in task_arrangements:
        # Create task sequence specific to this arrangement
        task_sequence = ContinualTaskSequence(
            dataset_name=base_config.dataset_name,
            task_type=task_type,
            n_tasks=base_config.num_tasks,
            root=base_config.dataset_root
        )
        
        # Create data loader
        data_loader = ContinualDataLoader(
            task_sequence=task_sequence,
            batch_size=base_config.batch_size,
            num_workers=base_config.num_workers
        )
        
        # Run experiment with this task arrangement
        result = run_cl_experiment(
            task_type=task_type,
            data_loader=data_loader,
            base_config=base_config,
            result_dir=base_config.results_dir
        )
        
        results[task_type] = result
    
    # Compare performance across task arrangements
    print("\nComparing task arrangement performances:")
    avg_accuracies = {arrangement: results[arrangement]['final_accuracies'].get('average', 0) 
                     for arrangement in task_arrangements}
    avg_forgetting = {arrangement: results[arrangement]['forgetting'].get('average', 0) 
                     for arrangement in task_arrangements}
    
    for arrangement in avg_accuracies:
        print(f"{arrangement}: Accuracy = {avg_accuracies[arrangement]:.4f}, Forgetting = {avg_forgetting[arrangement]:.4f}")
    
    # Visualize comparisons
    print("\nGenerating comparison visualizations...")
    
    # Final accuracy comparison
    plt.figure(figsize=(10, 6))
    arrangements = list(results.keys())
    accuracies = [avg_accuracies[arr] for arr in arrangements]
    plt.bar(arrangements, accuracies)
    plt.ylim(0, 1.0)
    plt.title('Average Accuracy by Task Arrangement')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', alpha=0.3)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.savefig(os.path.join(base_config.results_dir, "task_arrangement_accuracy_comparison.png"))
    
    # Forgetting comparison
    plt.figure(figsize=(10, 6))
    forgetting_values = [avg_forgetting[arr] for arr in arrangements]
    plt.bar(arrangements, forgetting_values)
    plt.title('Average Forgetting by Task Arrangement')
    plt.ylabel('Forgetting')
    plt.grid(axis='y', alpha=0.3)
    for i, f in enumerate(forgetting_values):
        plt.text(i, f + 0.01, f'{f:.4f}', ha='center')
    plt.savefig(os.path.join(base_config.results_dir, "task_arrangement_forgetting_comparison.png"))
    
    # Task-by-task accuracy comparison
    plt.figure(figsize=(12, 6))
    
    # For each task arrangement, plot task accuracies
    width = 0.35  # Width of bars
    task_keys = [k for k in results[arrangements[0]]['final_accuracies'] if k.startswith('task_')]
    task_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    task_indices = np.arange(len(task_keys))
    
    for i, arr in enumerate(arrangements):
        arr_accuracies = [results[arr]['final_accuracies'][task] for task in task_keys]
        plt.bar(task_indices + (i - 0.5) * width, arr_accuracies, width, label=arr)
    
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Task-by-Task Accuracy Comparison')
    plt.xticks(task_indices, task_keys)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(base_config.results_dir, "task_by_task_accuracy_comparison.png"))
    
    # Save individual arrangement plots
    for arr in results:
        viz.plot_task_performance(
            results[arr]['final_accuracies'],
            save_path=os.path.join(base_config.results_dir, f"{arr}_task_performance.png")
        )
        
        viz.plot_forgetting_curves(
            results[arr]['history'],
            save_path=os.path.join(base_config.results_dir, f"{arr}_forgetting_curves.png")
        )
    
    print(f"\nExperiment complete! Results saved to {base_config.results_dir}")

if __name__ == "__main__":
    main()