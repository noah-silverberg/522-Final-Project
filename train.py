import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from resnet import resnet20, resnet32, resnet44, resnet56, resnet110


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10_dataloaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
):
    """
    Returns train, val, and test dataloaders for CIFAR-10.
    """
    # CIFAR-10 statistics
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

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

    # Train set (will be split into train + val)
    full_train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Validation split
    num_train = len(full_train_dataset)
    num_val = int(val_fraction * num_train)
    num_train = num_train - num_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42),
    )

    # For val, we might prefer no augmentation, so wrap with a new dataset using test_transform
    val_dataset = Subset(
        datasets.CIFAR10(
            root=data_root,
            train=True,
            download=False,
            transform=test_transform,
        ),
        val_dataset.indices,
    )

    # Test set
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_and_evaluate_resnet(
    depth: int = 32,
    dropout_p: float = 0.0,
    seed: int = 42,
    batch_size: int = 128,
    num_epochs: int = 100,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    gamma: float = 0.1,
    step_size: int = 50,
    out_dir: str = "./checkpoints",
):
    """
    Train a CIFAR ResNet of a given depth with a specified dropout rate.
    Returns logs and test metrics.
    """
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=batch_size
    )

    # Choose model constructor based on depth
    if depth == 20:
        model = resnet20(num_classes=10, dropout_p=dropout_p)
    elif depth == 32:
        model = resnet32(num_classes=10, dropout_p=dropout_p)
    elif depth == 44:
        model = resnet44(num_classes=10, dropout_p=dropout_p)
    elif depth == 56:
        model = resnet56(num_classes=10, dropout_p=dropout_p)
    elif depth == 110:
        model = resnet110(num_classes=10, dropout_p=dropout_p)
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_acc = 0.0
    best_model_path = os.path.join(
        out_dir, f"cifar_resnet{depth}_drop{dropout_p}_seed{seed}.pth"
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "depth": depth,
                    "dropout_p": dropout_p,
                    "seed": seed,
                },
                best_model_path,
            )

    # Load best model for final test evaluation
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Final Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "checkpoint_path": best_model_path,
    }

