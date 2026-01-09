import logging
import os
from datetime import datetime
import sys
import torch
import numpy as np

def setup_logging(log_file=None, log_dir='./logs'):
    """Setup logging to both console and file"""
    # Create log directory if it doesn't exist
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    else:
        # Ensure the directory exists for the log file
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Create file handler (line buffered for immediate writes)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Set up root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger, log_file

# ==========================================
# 6. Checkpoint Saving
# ==========================================
def save_checkpoint(model, optimizer, epoch, test_acc, ckpt_dir, logger):
    """Save model checkpoint to .npz file"""
    ckpt_path = os.path.join(ckpt_dir, f"{epoch:010d}.npz")
    
    # Get model state dict (handle Opacus-wrapped models)
    if hasattr(model, '_module'):
        # Opacus wraps the model, get the underlying model's state dict
        model_state_dict = model._module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Get optimizer state dict
    optimizer_state_dict = optimizer.state_dict()
    
    # Convert to numpy arrays for .npz format
    # Store with prefixes to organize model_state_dict and optimizer_state_dict
    npz_dict = {}
    
    # Store model state dict with prefix
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            npz_dict[f'model_state_dict/{key}'] = value.detach().cpu().numpy()
        else:
            npz_dict[f'model_state_dict/{key}'] = value
    
    # Store optimizer state dict with prefix
    for key, value in optimizer_state_dict.items():
        if isinstance(value, torch.Tensor):
            npz_dict[f'optimizer_state_dict/{key}'] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            # Handle nested dicts in optimizer state (e.g., state for each parameter)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    npz_dict[f'optimizer_state_dict/{key}/{sub_key}'] = sub_value.detach().cpu().numpy()
                else:
                    npz_dict[f'optimizer_state_dict/{key}/{sub_key}'] = sub_value
        else:
            npz_dict[f'optimizer_state_dict/{key}'] = value
    
    # Save training metadata
    npz_dict['epoch'] = np.array([epoch], dtype=np.int32)
    npz_dict['test_accuracy'] = np.array([test_acc], dtype=np.float32)
    
    np.savez_compressed(ckpt_path, **npz_dict)
    logger.info(f"Checkpoint saved: {ckpt_path} (Epoch {epoch}, Test Accuracy: {test_acc:.2f}%)")

def find_latest_checkpoint(ckpt_dir):
    """Find the checkpoint file with the largest epoch number"""
    if not os.path.exists(ckpt_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.npz')]
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers from filenames (format: 0000000002.npz)
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.replace('.npz', ''))
            epochs.append((epoch, f))
        except ValueError:
            continue
    
    if not epochs:
        return None
    
    # Return the checkpoint with the largest epoch
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return os.path.join(ckpt_dir, latest_file), latest_epoch

def load_checkpoint(checkpoint_path, model, optimizer, device, logger=None):
    """Load model and optimizer state from .npz checkpoint file"""
    # Load the .npz file
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    
    # Extract epoch
    epoch = int(checkpoint['epoch'][0])
    
    # Reconstruct model state dict
    model_state_dict = {}
    for key in checkpoint.files:
        if key.startswith('model_state_dict/'):
            param_name = key.replace('model_state_dict/', '')
            # Convert numpy array back to torch tensor
            array = checkpoint[key]
            if isinstance(array, np.ndarray):
                model_state_dict[param_name] = torch.from_numpy(array).to(device)
            else:
                model_state_dict[param_name] = array
    
    # Reconstruct optimizer state dict
    # The optimizer state dict has structure: {'state': {...}, 'param_groups': [...]}
    optimizer_state_dict = {'state': {}, 'param_groups': []}
    
    # Collect all optimizer keys
    opt_data = {}
    for key in checkpoint.files:
        if key.startswith('optimizer_state_dict/'):
            # Remove prefix
            remaining = key.replace('optimizer_state_dict/', '')
            parts = remaining.split('/')
            
            if len(parts) == 1:
                # Top-level key (like 'param_groups')
                if parts[0] == 'param_groups':
                    # param_groups is a list, we'll handle it separately
                    opt_data['param_groups'] = checkpoint[key]
                else:
                    opt_data[parts[0]] = checkpoint[key]
            elif len(parts) == 2:
                # Nested: state/param_id or state/param_id/subkey
                if parts[0] == 'state':
                    param_id = int(parts[1]) if parts[1].isdigit() else parts[1]
                    if param_id not in opt_data:
                        opt_data[param_id] = {}
                    # This is a direct state value (shouldn't happen with our save format)
                    opt_data[param_id] = checkpoint[key]
            elif len(parts) == 3:
                # state/param_id/subkey (e.g., state/0/momentum_buffer)
                if parts[0] == 'state':
                    param_id = int(parts[1]) if parts[1].isdigit() else parts[1]
                    sub_key = parts[2]
                    if param_id not in optimizer_state_dict['state']:
                        optimizer_state_dict['state'][param_id] = {}
                    array = checkpoint[key]
                    if isinstance(array, np.ndarray):
                        optimizer_state_dict['state'][param_id][sub_key] = torch.from_numpy(array).to(device)
                    else:
                        optimizer_state_dict['state'][param_id][sub_key] = array
    
    # Handle param_groups if present
    if 'param_groups' in opt_data:
        optimizer_state_dict['param_groups'] = opt_data['param_groups'].tolist() if isinstance(opt_data['param_groups'], np.ndarray) else opt_data['param_groups']
    
    # Load model state dict (handle Opacus-wrapped models)
    if hasattr(model, '_module'):
        model._module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    
    # Load optimizer state dict
    try:
        optimizer.load_state_dict(optimizer_state_dict)
    except Exception as e:
        if logger:
            logger.warning(f"Could not fully load optimizer state: {e}. Continuing with default optimizer state.")
        else:
            print(f"Warning: Could not fully load optimizer state: {e}. Continuing with default optimizer state.")
    
    if logger:
        logger.info(f"Loaded checkpoint from epoch {epoch}")
    else:
        print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch