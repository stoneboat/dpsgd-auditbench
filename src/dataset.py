import torchvision
import torchvision.transforms as transforms
import torch


def get_data_loaders(data_dir='./data', logical_batch_size=4096, num_workers=2):
    """Create data loaders for CIFAR-10"""
    # We load "Clean" images. We will augment them K times inside the loop.
    # This saves memory (we don't load 4096*16 images at once).
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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