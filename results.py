import json
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# 1. Configuration & Style Setup
# ==========================================
RESULTS_DIR = 'results/mnist-updated'
OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def load_data(results_dir):
    """Loads and aggregates all JSON result files."""
    all_runs = []
    file_list = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not file_list:
        print(f"Error: No JSON files found in {results_dir}")
        return pd.DataFrame()

    print(f"Loading {len(file_list)} files...")
    for filename in file_list:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Handle structure: {dropout_rate: [list of runs]}
                for dropout_p, runs in data.items():
                    for run in runs:
                        run['dropout_p'] = float(dropout_p)
                        all_runs.append(run)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    df = pd.DataFrame(all_runs)
    
    # Calculate Generalization Gap (Train - Test)
    if 'train_acc' in df.columns and 'test_acc' in df.columns:
        df['gen_gap'] = df['train_acc'] - df['test_acc']
        
    # Drop failed runs
    required_cols = ['diffusion_curvature', 'ollivier_ricci', 'test_acc']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    
    print(f"Loaded {len(df)} valid runs.")
    return df

# ==========================================
# 2. Plotting Functions
# ==========================================

def plot_main_trends_row(df):
    """
    Creates a single row of 5 plots showing how metrics change with Dropout.
    Layout: [Test Acc] [Gen Gap] [Diff Curvature] [OR Curvature] [Loss Variance]
    """
    metrics = [
        ('test_acc', 'Test Accuracy (%)', '#1f77b4'),         # Blue
        ('gen_gap', 'Gen. Gap (%)', '#d62728'),               # Red
        ('diffusion_curvature', 'Diffusion Curv.', '#ff7f0e'), # Orange
        ('ollivier_ricci', 'Ollivier-Ricci', '#2ca02c'),      # Green (Added OR)
        ('loss_variance', 'Loss Variance', '#9467bd')         # Purple
    ]

    # Adjusted figsize to fit 5 columns comfortably
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), constrained_layout=True)
    
    for i, (col, label, color) in enumerate(metrics):
        if col not in df.columns:
            continue
            
        # Plot mean line with standard deviation band
        sns.lineplot(
            data=df, x='dropout_p', y=col, 
            marker='o', markersize=5, color=color, 
            errorbar='sd', ax=axes[i]
        )
        
        axes[i].set_xlabel("Dropout Rate ($p$)")
        axes[i].set_ylabel(label)
        axes[i].grid(True, linestyle='--')
        
        # Despine
        sns.despine(ax=axes[i])

    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_trends.pdf"))
    plt.close()
    print("Saved fig1_trends.pdf")

def plot_correlations(df):
    """
    Focuses on the mechanism: Does Curvature/Variance predict Generalization Gap?
    Plots: [Diff Curv vs Gap] [OR Curv vs Gap] [Loss Var vs Gap]
    """
    # 3 Subplots now
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    
    comparisons = [
        ('diffusion_curvature', 'Diffusion Curvature'),
        ('ollivier_ricci', 'Ollivier-Ricci Curvature'),  # Added OR
        ('loss_variance', 'Loss Variance')
    ]
    
    for i, (metric, label) in enumerate(comparisons):
        if metric not in df.columns:
            continue

        ax = axes[i]
        
        # Scatter plot colored by dropout rate
        scatter = ax.scatter(
            df[metric], df['gen_gap'], 
            c=df['dropout_p'], cmap='viridis', 
            s=40, alpha=0.8, edgecolors='w', linewidth=0.5
        )
        
        # Add regression line
        sns.regplot(
            data=df, x=metric, y='gen_gap', 
            scatter=False, ax=ax, color='gray', 
            line_kws={'linestyle': '--', 'linewidth': 1}
        )
        
        # Stats
        r, p = stats.pearsonr(df[metric], df['gen_gap'])
        
        text_str = f"$r = {r:.2f}$\n$p = {p:.2f}$"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9, bbox=props)
        
        ax.set_xlabel(label)
        if i == 0:
            ax.set_ylabel("Generalization Gap (%)")
        else:
            ax.set_ylabel("") # Remove y-label for 2nd and 3rd plots to save space
            
        ax.grid(True, linestyle='--', alpha=0.5)
        sns.despine(ax=ax)

    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Dropout Rate ($p$)')

    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_correlations.pdf"))
    plt.close()
    print("Saved fig2_correlations.pdf")

# ==========================================
# 3. LaTeX Table Generator
# ==========================================

def print_latex_table(df):
    """Generates the LaTeX code for the Results table."""
    
    # Define columns to aggregate
    cols_to_agg = {
        'test_acc': ['mean', 'std'],
        'gen_gap': ['mean', 'std'],
        'diffusion_curvature': ['mean', 'std'],
        'ollivier_ricci': ['mean', 'std'], # Added OR
        'loss_variance': ['mean', 'std']
    }
    
    # Filter only columns that exist
    cols_to_agg = {k: v for k, v in cols_to_agg.items() if k in df.columns}

    agg = df.groupby('dropout_p').agg(cols_to_agg).reset_index()

    print("\n" + "="*40)
    print("LATEX TABLE CODE")
    print("="*40)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{\textbf{Effect of Dropout on Generalization and Geometry.} Mean $\pm$ std over 10 seeds.}")
    print(r"\label{tab:main_results}")
    print(r"\resizebox{\textwidth}{%")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Dropout} ($p$) & \textbf{Test Acc} (\%) $\uparrow$ & \textbf{Gen Gap} (\%) $\downarrow$ & \textbf{Diff. Curv.} & \textbf{O-R Curv.} & \textbf{Loss Var.} \\")
    print(r"\midrule")

    for _, row in agg.iterrows():
        p = row['dropout_p'].values[0]
        
        def fmt(col, precision=2):
            if col not in cols_to_agg: return "N/A"
            m = row[col]['mean']
            s = row[col]['std']
            return f"{m:.{precision}f} $\pm$ {s:.{precision}f}"

        # 4 decimal places for curvature metrics usually looks better
        line = (f"{p:.1f} & {fmt('test_acc', 2)} & {fmt('gen_gap', 2)} & "
                f"{fmt('diffusion_curvature', 4)} & {fmt('ollivier_ricci', 4)} & {fmt('loss_variance', 4)} \\\\")
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")
    print("="*40 + "\n")

# ==========================================
# 4. Execution
# ==========================================

df = load_data(RESULTS_DIR)

if not df.empty:
    plot_main_trends_row(df)
    plot_correlations(df)
    print_latex_table(df)
else:
    print("No data found.")