# mnist_curvature_sweep.py

import os
import csv
import torch
from torch.utils.data import DataLoader

from data import get_data_loaders
from mnist import MLPMNIST
from curvature import get_loss_samples, compute_curvature

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where the MNIST checkpoints actually live in Drive
CHECKPOINT_ROOT = "/content/drive/MyDrive/third year/Geometry and topology in ML/final project/checkpoints"

# Where to save curvature results (also in Drive, same project folder)
RESULTS_DIR = "/content/drive/MyDrive/third year/Geometry and topology in ML/final project/curvature_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "mnist_curvature_sweep.csv")

# Dropout + epoch + seed pairs (using seed 11 for all)
MNIST_CONFIGS = [
    {"p": 0.0, "p_str": "0.0", "epoch": 10, "seed": 11},
    {"p": 0.1, "p_str": "0.1", "epoch": 15, "seed": 11},
    {"p": 0.2, "p_str": "0.2", "epoch": 15, "seed": 11},
    {"p": 0.3, "p_str": "0.3", "epoch": 20, "seed": 11},
    {"p": 0.4, "p_str": "0.4", "epoch": 20, "seed": 11},
]

# Alpha sets 
ALPHA_SETS = {
    "small": [1, 2, 3, 4, 5],
    "big": [10, 12, 14, 16, 18],
    "mixed": [1, 5, 10, 15, 20],
}

SAMPLES_PER_SCALE_VALUES = [10, 50, 100]


def load_checkpoint(dropout, epoch):
    """
    Adjust this to match Noah's filename pattern/output format.
    Example assumes something like:
      mnist_dropout_0.2_epoch15.pt
    """
    fname = f"mlpmnist_mnist_p{cfg['p_str']}_s{cfg['seed']}_epoch{cfg['epoch']}.pth"
    path = os.path.join(CHECKPOINT_ROOT, fname)
    print("Loading:", path)  # helpful debug line
    ckpt = torch.load(path, map_location=DEVICE)
    return ckpt


def build_model(dropout, state_dict):
    model = MLPMNIST(dropout=dropout).to(DEVICE)
    model.load_state_dict(state_dict)
    return model


def main():
    # Get loaders so we can evaluate loss on a held-out split
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name="mnist",
        batch_size=256,
        val_split=0.1,
        num_workers=2,
        data_root="./data",
    )
    eval_loader = val_loader

    # open CSV once and append all results
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dropout",
            "epoch",
            "alpha_set_name",
            "alphas",
            "samples_per_scale",
            "training_velocity",
            "diffusion_curvature",
            "ollivier_ricci",
            "loss_variance",
        ])

        for cfg in MNIST_CONFIGS:
            p = cfg["p"]
            epoch = cfg["epoch"]
            ckpt = load_checkpoint(cfg)


            state_dict = ckpt.get("model_state_dict", ckpt.get("model_state"))
            history = ckpt["history"]
            vel_idx = epoch - 1
            training_velocity = history["velocities"][vel_idx]
            
            model = build_model(p, state_dict)
            
            print(f"\n=== MNIST dropout={p}, epoch={epoch} (velocity={training_velocity:.4e}) ===")

            for alpha_name, alphas in ALPHA_SETS.items():
                for sps in SAMPLES_PER_SCALE_VALUES:
                    print(f"  -> alphas={alpha_name}={alphas}, samples_per_scale={sps}")

                    samples = get_loss_samples(
                        model=model,
                        data_loader=eval_loader,
                        training_velocity=training_velocity,
                        alphas=alphas,
                        samples_per_scale=sps,
                        device=DEVICE,
                    )

                    stats = compute_curvature(samples)

                    print(
                        f"     diffusion_curv={stats['diffusion_curvature']:.4f}, "
                        f"ollivier_ricci={stats['ollivier_ricci']:.4f}, "
                        f"loss_var={stats['loss_variance']:.4e}"
                    )

                    writer.writerow([
                        p,
                        epoch,
                        alpha_name,
                        repr(alphas),
                        sps,
                        float(training_velocity),
                        float(stats["diffusion_curvature"]),
                        float(stats["ollivier_ricci"]),
                        float(stats["loss_variance"]),
                    ])

    print(f"\nSaved all curvature results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
