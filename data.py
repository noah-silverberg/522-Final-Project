import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(dataset_name='cifar10', batch_size=128, val_split=0.1, num_workers=2, data_root='./data'):
    """
    Returns train and test dataloaders for CIFAR-10 or MNIST.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        # Standard CIFAR-10 stats
        mean = (0.49139968, 0.48215827, 0.44653124)
        std  = (0.24703233, 0.24348505, 0.26158768)
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    elif dataset_name == 'mnist':
        # MNIST is 1 channel
        mean = (0.13066062,)
        std  = (0.30810776,)
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=train_transform)
        test_set  = datasets.MNIST(root=data_root, train=False, download=True, transform=test_transform)
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    # Split training into train/val
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader