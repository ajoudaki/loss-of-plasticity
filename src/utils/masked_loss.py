import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    """
    A module that computes cross entropy loss while ignoring specified classes
    in both softmax and loss calculation.
    
    If no active classes are specified, works exactly like standard cross entropy.
    """
    
    def __init__(self, active_classes=None, reduction='mean'):
        """
        Args:
            active_classes: List or tensor of indices of active classes to include.
                            If None, all classes will be used (standard cross entropy).
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
        """
        super(MaskedCrossEntropy, self).__init__()
        self.active_classes = active_classes
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Forward pass for masked cross entropy loss.
        
        Args:
            logits: Raw model outputs [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
            
        Returns:
            Masked cross entropy loss or regular cross entropy if no active_classes specified
        """
        # If no active classes specified, use standard cross entropy
        if self.active_classes is None:
            return F.cross_entropy(logits, targets, reduction=self.reduction)
        
        device = logits.device
        
        # Convert active_classes to tensor if it's not already
        if not isinstance(self.active_classes, torch.Tensor):
            active_classes = torch.tensor(self.active_classes, device=device, dtype=torch.long)
        else:
            active_classes = self.active_classes.to(device)
            
        # Extract only the logits for active classes
        active_logits = logits[:, active_classes]
        
        # Create a mask for samples with active target classes
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for cls in active_classes:
            mask |= (targets == cls)
        
        if not torch.any(mask):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Select valid logits and targets
        valid_logits = active_logits[mask]
        valid_targets = targets[mask]
        
        # Create a mapping from original class indices to new positions
        mapping = torch.full((logits.size(1),), -1, dtype=torch.long, device=device)
        for i, cls in enumerate(active_classes):
            mapping[cls] = i
        
        # Map targets to new indices in the reduced space
        mapped_targets = mapping[valid_targets]
        
        # Compute cross entropy
        loss = F.cross_entropy(valid_logits, mapped_targets, reduction=self.reduction)
        
        return loss

    def extra_repr(self):
        return f'active_classes={self.active_classes}, reduction={self.reduction}'