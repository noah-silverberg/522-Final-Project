import torch

def evaluate_model(model, loader, criterion, device='cuda'):
    """
    Evaluates the model on (by default) ~10% of the dataset to save time
    for curvature / landscape computations.
    """
    model = model.to(device)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    # figure out how many batches this loader has
    try:
        num_batches = len(loader)
    except TypeError:
        num_batches = None

    # we'll use about 10% of the batches, but at least 1
    if num_batches is not None and num_batches > 0:
        max_batches = max(1, int(0.1 * num_batches))
    else:
        max_batches = None  # fall back to "use all" if we can't get len()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # stop early if we've hit our 10% batch budget
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / total
    
    print(f"Data Set (approx 10%) -> Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    return acc, avg_loss


def save_checkpoint(model, history, path):
    """
    Saves model state and training history.
    """
    state = {
        'model_state': model.state_dict(),
        'history': history
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

