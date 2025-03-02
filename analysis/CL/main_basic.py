"""
Basic continual learning experiment with MLP on CIFAR10.
This script demonstrates the core functionality of the framework.
"""

import os
import torch
import torch.nn as nn

from utils.config import ExperimentConfig, create_model_from_config, save_config
from data.datasets import ContinualTaskSequence, ContinualDataLoader
from training.trainer import ContinualTrainer
import utils.visualization as viz

def main():
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="mlp_cifar10_pairs",
        random_seed=42,
        dataset_name="cifar10",
        task_type="pairs",  # Each task contains a pair of classes
        num_tasks=5,        # 5 tasks of 2 classes each = all 10 CIFAR classes
        model_type="mlp",
        model_config={
            "hidden_sizes": [512, 256, 128],
            "activation": "relu",
            "dropout_p": 0.2,
        },
        optimizer_type="adam",
        optimizer_config={"lr": 0.001},
        batch_size=128,
        num_epochs_per_task=5,
        results_dir="./results/basic_experiment",
        checkpoint_dir="./checkpoints/basic_experiment",
    )
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(config.results_dir, "config.yaml"))
    
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
    print(f"Starting training on {config.num_tasks} tasks...")
    history = trainer.train_all_tasks(num_epochs_per_task=config.num_epochs_per_task)
    
    # Final evaluation on all tasks
    print("\nFinal evaluation on all tasks:")
    final_accuracies = trainer.evaluate_all_tasks()
    
    # Compute forgetting
    forgetting = trainer.compute_forgetting()
    print("\nForgetting metrics:")
    for task, value in forgetting.items():
        if task != "average":
            print(f"{task}: {value:.4f}")
    print(f"Average forgetting: {forgetting.get('average', 0):.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    # Plot task performance
    viz.plot_task_performance(
        final_accuracies,
        save_path=os.path.join(config.results_dir, "task_performance.png")
    )
    
    # Plot forgetting curves
    viz.plot_forgetting_curves(
        history,
        save_path=os.path.join(config.results_dir, "forgetting_curves.png")
    )
    
    print(f"\nExperiment complete! Results saved to {config.results_dir}")

if __name__ == "__main__":
    main()