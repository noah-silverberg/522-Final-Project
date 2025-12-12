import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data_loaders
from train import train_model
from mnist import MLPMNIST


def run_mnist_mlp_experiments(epochs=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Use your existing data loader helper
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name='mnist',
        batch_size=128,
        val_split=0.1,
        num_workers=2,
        data_root='./data',
    )

    dropout_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    all_results = []

    for p in dropout_values:
        print(f" MNIST MLP with dropout={p}")

        model = MLPMNIST(dropout=p)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler=None,
            epochs=epochs,
            device=device,
        )

        # evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, preds = logits.max(1)
                total += y.size(0)
                correct += preds.eq(y).sum().item()
        test_acc = 100.0 * correct / total

        # If you added best_epoch tracking to train_model, these keys exist.
        # Otherwise, you can comment these 3 lines out.
        best_epoch = history.get('best_epoch', None)
        best_val_acc = history['val_acc'][-1] if best_epoch is None else history['best_val_acc']
        best_val_loss = history.get('best_val_loss', history['val_loss'][-1])

        print(
            f"Dropout={p} | "
            f"Best val epoch={best_epoch} | "
            f"Best val acc={best_val_acc:.2f} | "
            f"Test acc={test_acc:.2f}"
        )

        all_results.append({
            "dropout": p,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "test_acc": test_acc,
        })

    return all_results


if __name__ == "__main__":
    run_mnist_mlp_experiments()
