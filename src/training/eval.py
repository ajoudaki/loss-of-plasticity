import torch
import torch.nn as nn
import torch.optim as optim

def evaluate_model(model, dataloader, criterion, device='cpu'):
    """
    Evaluate model on a dataset.
    
    Returns:
        loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total