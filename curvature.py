import torch
import torch.nn as nn
import numpy as np
from utils import evaluate_model
from scipy.spatial.distance import cdist
import scipy.sparse as sp
import ot

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

    with torch.no_grad():
        # Calculate loss at the exact minima (center) over dataset
        base_loss = evaluate_model(model, data_loader, criterion, device=device)[1]
        
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
                loss = evaluate_model(model, data_loader, criterion, device=device)[1]
                
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

def get_graph_operators(samples, alpha=1.0, mode='adaptive', fixed_sigma=1.0, k=10):
    """
    Converts the samples (point cloud) into a diffusion operator P and distance matrix D.
    
    Args:
        samples: List of dicts from get_loss_samples.
        alpha: Density normalization parameter (0.0 = Graph Laplacian, 1.0 = Laplace-Beltrami).
               Use 1.0 to remove the effect of sampling density.
        mode: Strategy for kernel bandwidth:
              - 'fixed': Use the provided 'fixed_sigma'.
              - 'heuristic': Use a single global sigma = median of k-NN distances.
              - 'adaptive': Use locally scaled sigmas (sigma_i = distance to k-th neighbor).
        fixed_sigma: Float value used only if mode='fixed'.
        k: Number of neighbors for sigma calculation (used in 'heuristic' and 'adaptive' modes).
        
    Returns:
        P: Row-normalized diffusion matrix (numpy array).
        D: Distance matrix (numpy array).
    """
    # 1. Extract perturbation vectors to form the point cloud matrix X
    X = np.vstack([s['perturbation'] for s in samples])
    
    # 2. Compute pairwise Euclidean distances
    D = cdist(X, X, metric='euclidean')
    
    # 3. Compute Kernel Matrix A based on mode (Bandwidth/Sigma step)
    if mode == 'fixed':
        sigma = fixed_sigma
        A = np.exp(-(D**2) / (sigma**2))
        
    elif mode == 'heuristic':
        sorted_dist = np.sort(D, axis=1)
        k = min(k, X.shape[0] - 1)
        knn_dists = sorted_dist[:, k]
        sigma = np.median(knn_dists)
        if sigma < 1e-12: sigma = 1.0 
        A = np.exp(-(D**2) / (sigma**2))
        
    elif mode == 'adaptive':
        sorted_dist = np.sort(D, axis=1)
        k = min(k, X.shape[0] - 1)
        sigmas = sorted_dist[:, k]
        sigmas[sigmas < 1e-12] = 1.0 
        sigma_outer = np.outer(sigmas, sigmas)
        A = np.exp(-(D**2) / sigma_outer)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 4. Density Normalization
    # If alpha > 0, we re-normalize the kernel to unbias for sampling density.
    # W_new(x,y) = W(x,y) / (q(x)^alpha * q(y)^alpha), where q(x) is the degree.
    if alpha > 0:
        # q(x): approximate density at each point
        q = np.sum(A, axis=1)
        
        if np.any(np.isclose(q, 0.0)):
            raise ValueError("Density q has near-zero entries, cannot normalize.")
        
        # Compute q^(-alpha)
        q_inv_alpha = np.power(q, -alpha)
        
        # Apply normalization: A_new = D^(-a) * A * D^(-a)
        # We use outer product to broadcast: term[i] * term[j]
        normalization_matrix = np.outer(q_inv_alpha, q_inv_alpha)
        A = A * normalization_matrix

    # 5. Row-Normalize to get Random Walk Matrix P
    row_sums = A.sum(axis=1, keepdims=True)
    if np.any(np.isclose(row_sums, 0.0)):
        raise ValueError("Row sums have near-zero entries, cannot normalize.")
    P = A / row_sums
    
    degrees = row_sums.flatten()

    return P, D, degrees

def compute_diffusion_curvature_metric(P, degrees, t=8, aperture=20, smoothing=True):
    """
    Calculates Diffusion Curvature using the spectral method.
    """
    N = P.shape[0]
    
    # --- 1. Spectral Powering (Using Similarity to Symmetric Ms) ---
    # P = D^(-1/2) * Ms * D^(1/2), where Ms is symmetric
    # Therefore:  P^t = D^(-1/2) * (V * Lambda^t * V.T) * D^(1/2)
    
    # Construct Ms = D^(1/2) * P * D^(-1/2)
    # Note: P = D^-1 A, so Ms = D^(-1/2) * A * D^(-1/2). 
    d_sqrt = np.sqrt(degrees)
    d_inv_sqrt = 1.0 / d_sqrt
    
    # Efficient broadcasting: Ms_ij = P_ij * d_i^(1/2) * d_j^(-1/2)
    Ms = P * d_sqrt[:, None] * d_inv_sqrt[None, :]
    
    # Ms should be symmetric
    if not np.allclose(Ms, Ms.T):
        raise ValueError("Matrix Ms is not symmetric.")
    
    # Eigendecomposition of Symmetric Matrix
    eigvals, V = np.linalg.eigh(Ms)
    
    # Power the eigenvalues
    eigvals_t = np.power(eigvals, t)
    
    # Reconstruct P^t
    # P^t = D^(-1/2) * (V * diag(vals^t) * V.T) * D^(1/2)
    # Intermediate: M_t = V * Lambda^t * V.T
    M_t = V @ (eigvals_t[:, None] * V.T)
    
    # Project back to P space
    P_powered = d_inv_sqrt[:, None] * M_t * d_sqrt[None, :]
    
    # --- 2. Thresholding (Replicating Paper Heuristic) ---
    # Set diffusion probability thresholds by sampling 100 points
    sample_indices = np.random.choice(N, size=100, replace=False)
    thresholds = []
    
    for idx in sample_indices:
        # Find the probability value of the 'aperture'-th neighbor
        # (The paper uses 'aperture' to define the size of the neighborhood B(x,r))
        row_data = P[idx, :] # Use P (transition probs), not P_powered, to define neighborhood
        thresh = np.partition(row_data, -aperture)[-aperture]
        thresholds.append(thresh)
    
    # Average threshold defines the "standard" neighborhood boundary probability
    P_threshold_val = np.mean(thresholds)
    
    # --- 3. Compute Laziness ---
    # Paper: "Sum of mass remaining in the initial neighborhood after t steps"
    
    # Identify the neighborhood mask (Where probability > threshold)
    # Note: The paper usually defines the neighborhood on the *original* graph P
    mask = (P >= P_threshold_val).astype(float)
    
    # Sum the *powered* diffusion mass landing in that neighborhood
    # C(x) = Sum_{y in Neighborhood} P^t(x, y)
    laziness = np.sum(P_powered * mask, axis=1)
    
    if smoothing:
        laziness = P @ laziness[:, None].flatten()
    
    return laziness[0] # Return value for the center point (index 0)

def compute_ollivier_ricci(P, D, num_neighbors=10):
    """
    Computes Ollivier-Ricci curvature at the center point.
    OR(x, y) = 1 - W_1(m_x, m_y) / d(x, y)
    
    We estimate scalar curvature at x by averaging OR(x, y) for all neighbors y.
    """
    ricci_sum = 0.0
    count = 0
    
    # Distribution at x (center is index 0)
    m_x = P[0, :].astype(np.float64)

    # Sort by transition probability
    neighbors_idx = np.argsort(P[0, :])[-num_neighbors:] # Top num_neighbors neighbors
    # Exclude self if present
    neighbors_idx = [n for n in neighbors_idx if n != 0]
    
    for n_idx in neighbors_idx:
        # Distribution at neighbor y
        m_y = P[n_idx, :].astype(np.float64)
        
        # Ground distance d(x, y)
        d_xy = D[0, n_idx]
        if np.isclose(d_xy, 0.0):
            continue  # Skip if distance is zero to avoid division by zero
            
        # Wasserstein distance W_1(m_x, m_y)
        # Cost matrix M is simply the distance matrix D
        wasserstein_dist = ot.emd2(m_x, m_y, D)
        
        # Ricci Curvature for edge (x, y)
        k_xy = 1 - (wasserstein_dist / d_xy)
        
        ricci_sum += k_xy
        count += 1
        
    return ricci_sum / count if count > 0 else 0.0

def compute_curvature(samples):
    # 1. Build Graph
    P, D_dist, degrees = get_graph_operators(samples, mode='adaptive', alpha=1.0)
    
    # 2. Compute Metrics
    diff_curv = compute_diffusion_curvature_metric(P, degrees, t=4, aperture=20, smoothing=True)
    
    # For Ollivier-Ricci, we need D_dist and P
    or_curv = compute_ollivier_ricci(P, D_dist, num_neighbors=10)
    
    # 3. Compute Loss Variance (simple statistic)
    losses = [s['loss'] for s in samples]
    
    return {
        'diffusion_curvature': diff_curv,
        'ollivier_ricci': or_curv,
        'loss_variance': np.var(losses)
    }