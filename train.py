import torch
import torch.nn as nn
import time
import copy
import tqdm
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=50, device='cuda'):
    """
    Trains the model with the provided optimizer, loss function, and scheduler.
    
    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: The loss function object (e.g., nn.CrossEntropyLoss()).
        optimizer: The optimizer object (e.g., torch.optim.SGD(...)).
        scheduler: (Optional) Learning rate scheduler object.
        epochs: Number of epochs to train.
        device: 'cuda' or 'cpu'.
        
    Returns:
        model: The trained model (final state, to analyze the specific minima reached).
        history: A dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    model = model.to(device)
    
    # Dictionary to track metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [],   'val_acc': [],
        'velocities': []
    }
    
    print(f"Starting training on {device} for {epochs} epochs...")
    start_time = time.time()

    # Initialize previous state variables
    prev_params = None
    prev_loss = None

    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        # We grab the current state as a flat vector
        current_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
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

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100. * correct / total

        # --- VELOCITY CALCULATION ---
        # Velocity = |Loss_t - Loss_{t-1}| / ||Params_t - Params_{t-1}||
        if prev_params is not None and prev_loss is not None:
            # Calculate L2 distance between current params (after update) and prev params
            # current_params is from START of epoch. We need the params NOW (END of epoch).
            # So we re-fetch the parameters.
            params_end_of_epoch = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            
            param_dist = torch.norm(params_end_of_epoch - current_params).item()
            loss_delta = abs(epoch_train_loss - prev_loss)
            
            if not np.isclose(param_dist, 0.0):
                velocity = loss_delta / param_dist
            else:
                velocity = 0.0
                
            history['velocities'].append(velocity)
        else:
            history['velocities'].append(0.0) # First epoch has no velocity
            
        # Update trackers for next iteration
        prev_loss = epoch_train_loss
        prev_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        # --- UPDATE HISTORY ---
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Print latest velocity if available
            v_str = f"{history['velocities'][-1]:.4f}" if history['velocities'] else "N/A"
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
                  f"Velocity: {v_str}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    
    return model, history