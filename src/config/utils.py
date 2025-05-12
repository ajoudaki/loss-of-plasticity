"""
Utilities for working with configurations and model setup.
"""
import os
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional, List

from typing import Optional
import torch
import logging

def get_device(device_str: Optional[str] = None, verbose: bool = False) -> torch.device:
    """
    Get the appropriate torch device, intelligently selecting the least utilized GPU 
    when multiple are present.
    
    Args:
        device_str: Device string (e.g., 'cuda', 'cuda:0', 'cpu', 'mps')
               If None, will select the best available device
        verbose: Whether to print information about the selection process
    
    Returns:
        torch.device: The selected device
    """
    logger = logging.getLogger(__name__)
    
    # If a specific device is requested, use it
    if device_str is not None:
        return torch.device(device_str)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        # If we have multiple GPUs, select the best one
        if torch.cuda.device_count() > 1:
            device = _select_best_gpu(verbose)
            if verbose:
                logger.info(f"Selected best GPU: {device}")
            return device
        else:
            return torch.device('cuda')
    # Check if MPS is available (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    # Fall back to CPU
    else:
        return torch.device('cpu')

def _select_best_gpu(verbose: bool = False) -> torch.device:
    """
    Select the best GPU based on memory usage and utilization.
    
    Args:
        verbose: Whether to print information about the GPU selection process
    
    Returns:
        torch.device: The selected GPU device
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Try to use pynvml for detailed GPU metrics
        import pynvml
        
        # Initialize NVML
        pynvml.nvmlInit()
        
        num_gpus = pynvml.nvmlDeviceGetCount()
        best_gpu_idx = 0
        best_score = 0
        
        for gpu_idx in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            
            # Get memory information
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = info.free
            total_memory = info.total
            memory_factor = free_memory / total_memory
            
            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            util_factor = (100 - gpu_util) / 100
            
            # Get process count (as a proxy for contention)
            proc_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            proc_count = len(proc_info)
            proc_factor = 1.0 / (1 + proc_count)  # More processes = lower score
            
            # Compute combined score (higher is better)
            # Weights: 60% memory, 30% utilization, 10% process count
            score = 0.6 * memory_factor + 0.3 * util_factor + 0.1 * proc_factor
            
            if verbose:
                device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                logger.info(f"GPU {gpu_idx} ({device_name}): Memory {memory_factor:.2%} free, "
                           f"Utilization {gpu_util}%, Processes {proc_count}, Score {score:.4f}")
            
            # Select GPU with best score
            if gpu_idx == 0 or score > best_score:
                best_gpu_idx = gpu_idx
                best_score = score
        
        # Shutdown NVML
        pynvml.nvmlShutdown()
        
        return torch.device(f'cuda:{best_gpu_idx}')
    
    except (ImportError, Exception) as e:
        # Fall back to PyTorch's built-in functions (more limited)
        if verbose:
            logger.info(f"NVML not available ({str(e)}), using PyTorch GPU selection")
        return _select_best_gpu_pytorch(verbose)

def _select_best_gpu_pytorch(verbose: bool = False) -> torch.device:
    """
    Select the best GPU based only on memory usage (PyTorch fallback).
    
    Args:
        verbose: Whether to print information about the selection process
    
    Returns:
        torch.device: The selected GPU device
    """
    logger = logging.getLogger(__name__)
    
    num_gpus = torch.cuda.device_count()
    best_gpu_idx = 0
    best_free_memory = 0
    
    for gpu_idx in range(num_gpus):
        # Get memory information
        total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_idx)
        free_memory = total_memory - allocated_memory
        
        if verbose:
            device_name = torch.cuda.get_device_name(gpu_idx)
            logger.info(f"GPU {gpu_idx} ({device_name}): Free Memory {free_memory/1024**3:.2f}GB")
        
        if gpu_idx == 0 or free_memory > best_free_memory:
            best_gpu_idx = gpu_idx
            best_free_memory = free_memory
    
    return torch.device(f'cuda:{best_gpu_idx}')

def setup_wandb(cfg: DictConfig) -> bool:
    """
    Setup weights & biases logging.
    
    Args:
        cfg: Configuration object
        
    Returns:
        bool: True if wandb was initialized, False otherwise
    """
    if cfg.use_wandb:
        # Prepare wandb config
        # Set resolve=False to avoid issues with special Hydra interpolations like hydra.job.num
        # W&B will log the config with unresolved interpolations as strings, which is acceptable.
        wandb_config = OmegaConf.to_container(cfg, resolve=False)
        
        # Create a descriptive run name with the requested parameters
        model_name = cfg.model.name
        if hasattr(cfg, "dataset"):
            dataset_name = cfg.dataset.name
        else:
            dataset_name = "N/A"
        
        if hasattr(cfg, "training"):
            training_type = cfg.training.training_type
        else:
            training_type = "N/A"
        
        # Initialize wandb with optional entity parameter and the created run name
        init_args = {
            "project": cfg.wandb_project,
            "tags": cfg.wandb_tags + [training_type], 
            "config": wandb_config,
        }

        # Add entity parameter if it exists
        if hasattr(cfg.logging, "wandb_entity"):
            init_args["entity"] = cfg.logging.wandb_entity

        
        # Add entity parameter if it exists
        if hasattr(cfg.logging, "wandb_run_name"):
            init_args["name"] = cfg.logging.wandb_run_name
            
        wandb.init(**init_args)
        return True
    return False

def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.

    Args:
        model: The model whose parameters will be optimized
        cfg: Configuration object containing optimizer settings

    Returns:
        Optimizer instance
    """
    optimizer_name = cfg.optimizer.name.lower()

    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            dampening=cfg.optimizer.dampening,
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=cfg.optimizer.nesterov
        )
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.optimizer.lr,
            alpha=cfg.optimizer.alpha,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=cfg.optimizer.momentum,
            centered=cfg.optimizer.centered
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == 'noisysgd':
        from ..utils.noisy_optimizer import NoisySGD
        return NoisySGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            noise_scale=cfg.optimizer.noise_scale,
            noise_decay=cfg.optimizer.noise_decay,
            momentum=cfg.optimizer.momentum,
            dampening=cfg.optimizer.dampening,
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=cfg.optimizer.nesterov
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")