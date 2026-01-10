import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def compute_loss(model, canaries, device, batch_size=128):
    """
    Computes CrossEntropyLoss for a set of canaries (images) against model w.
    """
    model.eval()  # Set to evaluation mode (disable dropout, etc.)
    
    # Define loss function with reduction='none' to get per-sample losses
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Convert dataset to DataLoader if needed
    if isinstance(canaries, torch.utils.data.Dataset):
        canary_loader = DataLoader(canaries, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        canary_loader = canaries
    
    all_losses = []
    
    with torch.no_grad():
        for images, targets in canary_loader:
            # Move inputs to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute per-sample losses
            losses = criterion(outputs, targets)
            
            # Store losses
            all_losses.append(losses.cpu().numpy())
    
    # Concatenate all losses into a single array
    all_losses = np.concatenate(all_losses)
    
    return all_losses


def compute_audit_score_blackbox(initial_model, final_model, canaries, device, batch_size=128):
    """
    Computes audit scores for a set of canaries.
    
    Returns:
        numpy.ndarray: Per-sample audit scores, where score = l(w_0) - l(w_l)
                      If initial_model is None, returns -loss_final (negative final loss)
    """
    loss_final = compute_loss(final_model, canaries, device, batch_size=batch_size)
    
    if initial_model is not None:
        loss_initial = compute_loss(initial_model, canaries, device, batch_size=batch_size)
    else:
        # Fallback if w_0 is lost: Score is just negative final loss
        loss_initial = np.zeros_like(loss_final)
    scores = loss_initial - loss_final   
    return scores