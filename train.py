import torch
import torch.nn as nn
import time
import tqdm
import numpy as np
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                epochs=50, device='cuda', save_path=None):
    """
    Trains the model. If save_path exists, loads that model instead of training.
    
    Args:
        save_path: Path to save/load model checkpoint. If None, no saving/loading.
    """
    
    # Check if we already trained this model
    if save_path and os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return model, checkpoint['history']
    
    # Otherwise, train as normal
    model = model.to(device)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'velocities': []
    }
    
    print(f"Starting training on {device} for {epochs} epochs...")
    start_time = time.time()

    prev_params = None
    prev_loss = None

    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
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
        if prev_params is not None and prev_loss is not None:
            params_end_of_epoch = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            param_dist = torch.norm(params_end_of_epoch - current_params).item()
            loss_delta = abs(epoch_train_loss - prev_loss)
            velocity = loss_delta / param_dist if not np.isclose(param_dist, 0.0) else 0.0
            history['velocities'].append(velocity)
        else:
            history['velocities'].append(0.0)
            
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
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        if scheduler:
            scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            v_str = f"{history['velocities'][-1]:.4f}" if history['velocities'] else "N/A"
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
                  f"Velocity: {v_str}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    
    # Save after training
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, save_path)
        print(f"Model saved to {save_path}")
    
    return model, history
