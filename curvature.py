import torch
import torch.nn as nn
import numpy as np
import copy

def vector_to_parameters(vec, parameters):
    """
    Overwrites the parameters of the model with the values in 'vec'.
    Inverse of torch.nn.utils.parameters_to_vector.
    """
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data = vec[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

def get_loss_samples(model, data_loader, training_velocity, alphas=[1.0, 5.0, 10.0], samples_per_scale=50, device='cuda'):
    """
    Samples the loss landscape using the Concentric Shell Sampling strategy (from Diffusion Curvature paper).
    
    It samples points on hyperspheres (shells) around the current parameters. 
    The radius of these shells is determined by the 'training_velocity' and 'alphas'.
    
    Args:
        model: The trained PyTorch model.
        data_loader: DataLoader to evaluate loss.
        training_velocity (float): The rate of change of loss w.r.t parameter distance (dLoss/dParam). 
                                   This adapts the sampling scale to the model's current "speed" in the landscape.
        alphas (list of floats): A list of multipliers. Radius = training_velocity * alpha.
        samples_per_scale (int): Number of points to sample for each alpha (shell).
        device: 'cuda' or 'cpu'.
        
    Returns:
        samples: A list of dictionaries, each containing:
            - 'perturbation': The random vector added (numpy array).
            - 'loss': The loss value at that perturbed point.
            - 'alpha': The alpha scalar used for this shell.
            - 'radius': The Euclidean norm of the perturbation.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Flatten current weights to a single vector (the "center")
    original_params_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    results = []
    
    # Determine the list of radii to sample
    radii = [training_velocity * a for a in alphas]
    
    # Helper to calculate loss over the loader
    def evaluate_loss_on_loader(m, loader):
        total_loss = 0.0
        total_count = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = m(inputs)
            loss = criterion(outputs, targets)
            # accumulate sum of losses
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)
        return total_loss / total_count if total_count > 0 else 0.0

    with torch.no_grad():
        # Calculate loss at the exact minima (center) over dataset
        base_loss = evaluate_loss_on_loader(model, data_loader)
        
        # Record the center point (radius 0)
        results.append({
            'perturbation': np.zeros(original_params_vec.shape[0]), 
            'loss': base_loss,
            'alpha': 0.0,
            'radius': 0.0
        })

        # Loop over each "shell" (alpha/radius)
        for r, alpha in zip(radii, alphas):
            for i in range(samples_per_scale):
                # 2. Generate random direction on the hypersphere
                # Draw from standard normal
                u = torch.randn_like(original_params_vec)
                
                # Normalize to unit length (direction vector)
                norm = torch.norm(u)

                while norm < 1e-12: # repeat sampling if too close to zero (for numerics)
                    u = torch.randn_like(original_params_vec)
                    norm = torch.norm(u)

                direction = u / norm
                
                # Scale by the specific radius for this shell
                perturbation = direction * r
                
                # 3. Apply perturbed weights to model
                perturbed_params = original_params_vec + perturbation
                vector_to_parameters(perturbed_params, model.parameters())
                
                # 4. Compute Loss
                loss = evaluate_loss_on_loader(model, data_loader)
                
                # Store result
                results.append({
                    'perturbation': perturbation.cpu().numpy(), 
                    'loss': loss,
                    'alpha': alpha,
                    'radius': r
                })
            
    # 5. Restore original weights
    vector_to_parameters(original_params_vec, model.parameters())
    
    return results

def compute_curvature(samples):
    raise NotImplementedError("Curvature computation not yet implemented.")