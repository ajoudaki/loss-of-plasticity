import torch
from collections import defaultdict

class NetworkMonitor:
    def __init__(self, model, filter_func=None):
        """
        Initialize the network monitor.
        
        Args:
            model: The neural network model to monitor
            filter_func: Function that takes a layer name and returns 
                         True if the layer should be monitored
        """
        self.model = model
        self.filter_func = filter_func if filter_func is not None else lambda name: True
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.hooks_active = False
        
    def set_filter(self, filter_func):
        """Update the filter function for selecting layers to monitor."""
        was_active = self.hooks_active
        if was_active:
            self.remove_hooks()
        self.filter_func = filter_func if filter_func is not None else lambda name: True
        if was_active:
            self.register_hooks()
        
    def register_hooks(self):
        """Register forward and backward hooks on the model."""
        if not self.hooks_active:
            for name, module in self.model.named_modules():
                if name != '' and self.filter_func(name):
                    def make_fwd_hook(name=name):
                        def hook(module, input, output):
                            self.activations[f"{name}"].append(output.clone().detach().cpu())
                        return hook
                    
                    def make_bwd_hook(name=name):
                        def hook(module, grad_input, grad_output):
                            if len(grad_output) > 0 and grad_output[0] is not None:
                                self.gradients[f"{name}"].append(grad_output[0].clone().detach().cpu())
                            return grad_input
                        return hook
                    
                    h1 = module.register_forward_hook(make_fwd_hook())
                    h2 = module.register_full_backward_hook(make_bwd_hook())
                    self.fwd_hooks.append(h1)
                    self.bwd_hooks.append(h2)
            
            self.hooks_active = True
    
    def remove_hooks(self):
        """Remove all hooks from the model."""
        if self.hooks_active:
            for h in self.fwd_hooks + self.bwd_hooks:
                h.remove()
            self.fwd_hooks = []
            self.bwd_hooks = []
            self.hooks_active = False
        
    def clear_data(self):
        """Clear stored activations and gradients."""
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)
        
    def get_latest_activations(self):
        """Get the latest activations for all monitored layers."""
        latest_acts = {}
        for name, acts_list in self.activations.items():
            if acts_list:
                latest_acts[name] = acts_list[-1]
        return latest_acts
    
    def get_latest_gradients(self):
        """Get the latest gradients for all monitored layers."""
        latest_grads = {}
        for name, grads_list in self.gradients.items():
            if grads_list:
                latest_grads[name] = grads_list[-1]
        return latest_grads
