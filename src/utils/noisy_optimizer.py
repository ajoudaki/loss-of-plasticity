"""
Noisy Backpropagation optimizer used for continual learning.

This optimizer adds noise to the gradients during backpropagation,
which can help overcome local minima and maintain model plasticity.
"""
import torch
from torch.optim import SGD
from typing import List, Dict, Any, Optional, Union, Callable


class NoisySGD(SGD):
    """
    Noisy SGD optimizer that adds Gaussian noise to gradients during optimization.
    
    This optimizer is often used in continual learning to help maintain plasticity
    by allowing the model to avoid getting stuck in sub-optimal local minima.
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate
        noise_scale: Scale of the noise to add (relative to the gradient norm)
        noise_decay: Decay rate for the noise scale over time
        momentum: Momentum factor for SGD
        dampening: Dampening for momentum
        weight_decay: Weight decay for L2 regularization
        nesterov: Whether to use Nesterov momentum
    """
    def __init__(self, params, lr=0.01, noise_scale=0.01, noise_decay=1.0, 
                 momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.steps = 0
        
    def reset_scale(self, new_scale: float=None):
        """
        Resets the noise scale to a new value.
        
        Args:
            new_scale: The new scale for the noise
        """
        if new_scale is not None:
            self.noise_scale = new_scale
        else:
            self.noise_scale = 1
        self.steps = 0
        
    def step(self, closure=None):
        """
        Performs a single optimization step with added noise.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Calculate current noise scale with decay
        current_noise_scale = self.noise_scale * self.noise_decay ** self.steps
        self.steps += 1
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Get gradient norm for scaling the noise appropriately
                grad_norm = d_p.norm().item()
                
                # Add Gaussian noise scaled relative to gradient norm
                if grad_norm > 0 and current_noise_scale > 0:
                    noise = torch.randn_like(d_p) * grad_norm * current_noise_scale
                    d_p.add_(noise)
                
                # Handle weight decay, momentum, etc. exactly as in regular SGD
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                p.data.add_(d_p, alpha=-group['lr'])
        
        return loss