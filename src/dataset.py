import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset, ConcatDataset

def get_data_loaders(data_dir, logical_batch_size, num_workers):
    """Create data loaders for CIFAR-10"""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=logical_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    return train_loader, test_dataset

# ==========================================
# 1. Canary Transformation Function
# ==========================================
def make_mislabeled_canaries(dataset, seed=42):
    """
    Takes a dataset (subset of real images) and turns them into a canary set
    by randomly relabeling them (Input Space Attack - Mislabeled).
    """
    # Create a shallow copy to avoid modifying the original dataset in place if shared
    canary_dataset = copy.copy(dataset)
    
    # Access the underlying data (assuming CIFAR10 structure)
    # If dataset is a Subset, we need to access the underlying dataset's targets via indices
    if isinstance(dataset, Subset):
        indices = dataset.indices
        original_targets = np.array(dataset.dataset.targets)[indices]
    else:
        indices = np.arange(len(dataset))
        original_targets = np.array(dataset.targets)

    # Set seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate new labels ensuring new_label != old_label
    # We add a random integer [1, 9] to the current label modulo 10
    shifts = rng.integers(1, 10, size=len(indices))
    new_targets = (original_targets + shifts) % 10

    # We need a custom wrapper to override the __getitem__ target
    class MislabeledWrapper(torch.utils.data.Dataset):
        def __init__(self, original_subset, new_labels):
            self.dataset = original_subset
            self.new_labels = new_labels

        def __getitem__(self, idx):
            img, _ = self.dataset[idx]
            return img, self.new_labels[idx]

        def __len__(self):
            return len(self.dataset)

    return MislabeledWrapper(canary_dataset, new_targets)

# ==========================================
# 2. Canary Mask Generation Function
# ==========================================
def generate_canary_mask(num_canaries, pkeep=0.5, seed=42):
    """
    Takes the size of the canary set and outputs a boolean mask indicating 
    picking a canary or not.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(size=num_canaries) < pkeep
    
    return mask

# ==========================================
# 3. Canary Generation Function
# ==========================================
def generate_poisoned_canaries_and_mask(
    data_dir='./data', 
    canary_count=500,     
    seed=42,
    pkeep=0.5
):
    """
    Generates poisoned canaries and inclusion mask for privacy auditing.
    
    Args:
        data_dir: Directory for CIFAR-10 data
        canary_count: Number of canaries to create
        seed: Random seed for reproducibility
        pkeep: Probability of including each canary in the training set
    
    Returns:
        poisoned_canaries: Dataset with mislabeled canaries
        inclusion_mask: Boolean mask indicating which canaries are included
    """
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load Full Training Dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # ---------------------------------------------------------
    # AUDITING LOGIC
    # ---------------------------------------------------------
    total_len = len(full_train_dataset)
    indices = np.arange(total_len)
    
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    canary_indices = indices[:canary_count]
    
    canary_candidates = Subset(full_train_dataset, canary_indices)
    poisoned_canaries = make_mislabeled_canaries(canary_candidates, seed=seed)
    inclusion_mask = generate_canary_mask(canary_count, pkeep=pkeep, seed=seed)
    
    return poisoned_canaries, inclusion_mask

# ==========================================
# 4. Auditable Data Loader Function
# ==========================================
def get_auditable_data_loaders(
    data_dir='./data', 
    canary_count=500,     
    seed=42,
    pkeep=0.5
):
    """
    Creates data loaders for CIFAR-10 with integrated privacy auditing.
    
    Logic:
    1. Loads Training Data.
    2. Splits into 'Canary Candidates' (m) and 'Regular Data' (n-m)[cite: 53].
    3. Transforms candidates into Mislabeled Canaries[cite: 618].
    4. Applies random mask (coin flips) to include/exclude canaries[cite: 55].
    5. Returns Combined Train Loader and Clean Test Loader.
    """
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load Full Datasets
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Generate poisoned canaries and inclusion mask
    poisoned_canaries, inclusion_mask = generate_poisoned_canaries_and_mask(
        data_dir=data_dir,
        canary_count=canary_count,
        seed=seed,
        pkeep=pkeep
    )
    
    # Reconstruct indices to get regular dataset (same seed ensures same shuffle)
    total_len = len(full_train_dataset)
    indices = np.arange(total_len)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    regular_indices = indices[canary_count:]
    
    # Combine regular dataset with selected canaries
    regular_dataset = Subset(full_train_dataset, regular_indices)
    kept_canary_indices = np.where(inclusion_mask)[0]
    final_canary_dataset = Subset(poisoned_canaries, kept_canary_indices)
    final_train_dataset = ConcatDataset([regular_dataset, final_canary_dataset])
    
    return final_train_dataset, test_dataset

