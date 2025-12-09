import torch
import torch.nn as nn
import numpy as np
import os
import json

# Import your modules
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from data import get_data_loaders
from train import train_model
from utils import evaluate_model
from curvature import get_loss_samples, compute_curvature

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(dataset='cifar10', network='resnet110', epochs=200, seeds=[36, 84, 68, 79, 11, 82, 77, 31, 26, 18], dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.4], alphas=[1., 3., 5., 7., 10., 13., 16., 20.], samples_per_scale=100, skip_curvature=False, results_filename='experiment_results.json'):
    """
    Runs the full experiment: 
    1. Loop over dropout rates.
    2. Loop over random seeds.
    3. Train -> Sample Curvature -> Save.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {} # To store all metrics
    
    # Create directory for saved models/results
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print(f"Running Experiment on {dataset} with model {network} for {epochs} epochs, dropouts {dropout_rates}, alphas {alphas}, and samples per scale {samples_per_scale}")

    for p in dropout_rates:
        results[p] = []
        print(f"\n=== Testing Dropout Rate: {p} ===")
        
        for seed in seeds:
            print(f"--- Seed {seed} ---")
            set_seed(seed)
            
            # Load Data
            train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size=128)
            
            # Initialize Model
            if network == 'resnet20':
                model = resnet20(num_classes=10, in_channels=3 if dataset == 'cifar10' else 1, p=p)
            elif network == 'resnet32':
                model = resnet32(num_classes=10, in_channels=3 if dataset == 'cifar10' else 1, p=p)
            elif network == 'resnet44':
                model = resnet44(num_classes=10, in_channels=3 if dataset == 'cifar10' else 1, p=p)
            elif network == 'resnet56':
                model = resnet56(num_classes=10, in_channels=3 if dataset == 'cifar10' else 1, p=p)
            elif network == 'resnet110':
                model = resnet110(num_classes=10, in_channels=3 if dataset == 'cifar10' else 1, p=p)

            # Setup Training Objects
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
            criterion = nn.CrossEntropyLoss()
            
            # Train
            ckpt_name = f"checkpoints/{network}_{dataset}_p{p}_s{seed}"
            model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=epochs, device=device, save_path=ckpt_name)
            
            # Get the velocity from the final epoch of training
            final_velocity = history['velocities'][-1]
            if np.isclose(final_velocity, 0.0):
                raise ValueError("Final training velocity is zero, cannot proceed with curvature sampling.")
            
            print(f"Dynamic Training Velocity Calculated: {final_velocity:.4f}")
            
            # Evaluate Generalization
            acc, test_loss = evaluate_model(model, test_loader, criterion, device=device)
            
            if skip_curvature:
                continue

            # Analyze Curvature
            print("Sampling curvature (this may take a while)...")
            # Note: We pass 'train_loader' to evaluate curvature on the training set landscape.
            samples = get_loss_samples(
                model, 
                train_loader, 
                training_velocity=final_velocity,
                alphas=alphas, 
                samples_per_scale=samples_per_scale, 
                device=device
            )
            
            curvature_metrics = compute_curvature(samples)
            
            # Record Data
            run_data = {
                'seed': seed,
                'dropout_p': p,
                'test_acc': acc,
                'test_loss': test_loss,
                'diffusion_curvature': curvature_metrics['diffusion_curvature'],
                'ollivier_ricci': curvature_metrics['ollivier_ricci'],
                'loss_variance': curvature_metrics['loss_variance']
            }
            results[p].append(run_data)
            
            print(f"Result: Acc={acc:.2f}%, DiffCurve={run_data['diffusion_curvature']:.4e}, ORCurve={run_data['ollivier_ricci']:.4e}, LossVar={run_data['loss_variance']:.4e}")

    # Save full results
    with open(f'results/{results_filename}', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExperiments Complete. Results saved to results/{results_filename}")

if __name__ == "__main__":
    run_experiment(dataset='cifar10', epochs=20, seeds=[42])