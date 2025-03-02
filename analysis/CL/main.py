# main.py - Example experiment script

import torch
from utils.config import ExperimentConfig, create_model_from_config, save_config
from data.datasets import ContinualTaskSequence, ContinualDataLoader
from training.trainer import ContinualTrainer
from analysis.activation_analysis import analyze_activations
from analysis.rank_analysis import analyze_rank_dynamics
import utils.visualization as viz
import os

def main():
    # 1. Create or load experiment configuration
    config = ExperimentConfig(
        experiment_name="mlp_cifar10_pairs",
        dataset_name="cifar10",
        task_type="pairs",
        num_tasks=5,  # 5 pairs of classes from CIFAR-10
        model_type="mlp",
        model_config={
            "hidden_sizes": [512, 256, 128],
            "activation": "relu",
            "normalization": "batch"
        },
        optimizer_type="adam",
        optimizer_config={"lr": 0.001, "weight_decay": 1e-4},
        batch_size=128,
        num_epochs_per_task=10,
        metrics_config={
            "compute_rank_metrics": True,
            "compute_component_metrics": True,
            "record_activations": True
        }
    )
    
    # Create directories for results
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Save the configuration
    save_config(config, os.path.join(config.results_dir, f"{config.experiment_name}_config.yaml"))
    
    # 2. Set up the dataset and tasks
    task_sequence = ContinualTaskSequence(
        dataset_name=config.dataset_name,
        task_type=config.task_type,
        n_tasks=config.num_tasks
    )
    
    data_loader = ContinualDataLoader(
        task_sequence=task_sequence,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # 3. Create the model
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    model = create_model_from_config(config)
    model.to(device)
    
    # 4. Define optimizer factory function
    def optimizer_factory(model):
        if config.optimizer_type == "adam":
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

    # 5. Set up the trainer
    trainer = ContinualTrainer(
        model=model,
        continual_data_loader=data_loader,
        optimizer_factory=optimizer_factory,
        device=device,
        metrics_config=config.metrics_config,
        checkpoint_dir=config.checkpoint_dir,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project
    )
    
    # 6. Train on all tasks
    history = trainer.train_all_tasks(num_epochs_per_task=config.num_epochs_per_task)
    
    # 7. Evaluate on all tasks
    final_accuracies = trainer.evaluate_all_tasks()
    forgetting = trainer.compute_forgetting()
    
    # 8. Visualize results
    viz.plot_task_performance(
        final_accuracies, 
        save_path=os.path.join(config.results_dir, f"{config.experiment_name}_task_perf.png")
    )
    
    viz.plot_forgetting_curves(
        history, 
        save_path=os.path.join(config.results_dir, f"{config.experiment_name}_forgetting.png")
    )
    
    # 9. Optional: Perform deeper analysis on the trained model
    # Analyze activations for a specific task
    sample_loader = data_loader.get_task_loader(task_idx=0, train=False)
    batch = next(iter(sample_loader))
    inputs = batch[0].to(device)
    
    # Record activations
    with torch.no_grad():
        if hasattr(model, 'record_activations'):
            model.record_activations = True
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs, activations = outputs
        else:
            activations = model.get_activations()
    
    # Analyze and visualize activations
    if activations:
        viz.plot_activation_heatmap(
            activations,
            layer_name='linear_0' if 'linear_0' in activations else list(activations.keys())[1],
            save_path=os.path.join(config.results_dir, f"{config.experiment_name}_activations.png")
        )

if __name__ == "__main__":
    main()