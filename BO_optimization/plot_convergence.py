#!/usr/bin/env python
"""Generate clean convergence plot with initial sampling"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_convergence(log_dir, max_iter=86):
    """Generate convergence plot showing initial + BO iterations"""

    # Load BO iteration data (1 to max_iter)
    files = sorted(glob.glob(f'{log_dir}/iter_*.json'))
    files = [f for f in files if int(f.split('_')[-1].replace('.json', '')) <= max_iter]

    bo_iterations = []
    bo_cvars = []

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            bo_iterations.append(data['iteration'])
            bo_cvars.append(data['cvar'])

    bo_iterations = np.array(bo_iterations)
    bo_cvars = np.array(bo_cvars)

    # Initial sampling data (from SESSION_26_SUMMARY.md)
    initial_cvars = [0.5852, 0.4229, 0.5503, 0.4811, 0.4422,
                     0.3339, 0.3275, 0.2915, 0.5522, 0.3115]
    initial_iters = np.arange(-9, 1)  # -9 to 0

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot initial sampling
    ax.plot(initial_iters, initial_cvars, 'o-', color='gray',
            linewidth=2, markersize=8, alpha=0.7, label='Initial Sampling (Random)')

    # Plot BO iterations
    ax.plot(bo_iterations, bo_cvars, 'o-', color='blue',
            linewidth=2, markersize=6, alpha=0.8, label='BO Iterations')

    # Cumulative best line
    all_cvars = initial_cvars + list(bo_cvars)
    all_iters = list(initial_iters) + list(bo_iterations)
    cumulative_best = np.maximum.accumulate(all_cvars)
    ax.plot(all_iters, cumulative_best, '--', color='red',
            linewidth=2, alpha=0.7, label='Cumulative Best')

    # Mark convergence region (81-86)
    convergence_start = 81
    convergence_end = 86
    ax.axvspan(convergence_start, convergence_end,
               alpha=0.2, color='green', label='Convergence Region')

    # Mark best iteration
    best_idx = np.argmax(bo_cvars)
    best_iter = bo_iterations[best_idx]
    best_cvar = bo_cvars[best_idx]
    ax.scatter([best_iter], [best_cvar], color='red', s=300,
               marker='*', zorder=5, edgecolors='black', linewidths=2)
    ax.text(best_iter, best_cvar + 0.01, f'Best: {best_cvar:.4f}\n(Iter {best_iter})',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Vertical line separating initial and BO
    ax.axvline(0.5, color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, 'BO Start',
            ha='center', va='top', fontsize=10, rotation=0,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and title
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('CVaR (α=0.3)', fontsize=14, fontweight='bold')
    ax.set_title('CVaR Optimization: Initial Sampling + BO Convergence\n'
                 f'Improvement: {initial_cvars[0]:.4f} → {best_cvar:.4f} '
                 f'(+{((best_cvar - initial_cvars[0]) / initial_cvars[0] * 100):.1f}%)',
                 fontsize=16, fontweight='bold', pad=20)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Set x-axis limits
    ax.set_xlim(-10, max_iter + 2)

    # Add statistics box
    stats_text = (
        f'Initial Best: {max(initial_cvars):.4f}\n'
        f'BO Start: {bo_cvars[0]:.4f}\n'
        f'Final Best: {best_cvar:.4f}\n'
        f'Convergence: Iter {convergence_start}-{convergence_end}\n'
        f'Total Iterations: {len(initial_cvars)} + {len(bo_cvars)}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')

    # Save
    output_path = f'results/convergence_plot_{log_dir.split("/")[-1]}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')

    # Print summary
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    print(f"Initial Sampling:")
    print(f"  Range: [{min(initial_cvars):.4f}, {max(initial_cvars):.4f}]")
    print(f"  Best: {max(initial_cvars):.4f}")
    print(f"\nBO Optimization (Iter 1-{max_iter}):")
    print(f"  Start CVaR: {bo_cvars[0]:.4f}")
    print(f"  Best CVaR: {best_cvar:.4f} (Iter {best_iter})")
    print(f"  Final CVaR: {bo_cvars[-1]:.4f} (Iter {max_iter})")
    print(f"\nConvergence Region (Iter {convergence_start}-{convergence_end}):")
    conv_cvars = bo_cvars[convergence_start-1:convergence_end]
    print(f"  Mean: {conv_cvars.mean():.4f} ± {conv_cvars.std():.4f}")
    print(f"  Range: [{conv_cvars.min():.4f}, {conv_cvars.max():.4f}]")
    print(f"\nTotal Improvement:")
    print(f"  {max(initial_cvars):.4f} → {best_cvar:.4f}")
    print(f"  +{((best_cvar - max(initial_cvars)) / max(initial_cvars) * 100):.1f}%")
    print("="*60)

    # plt.show()  # Commented out for automation

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "logs/run_20251120_151025"

    plot_convergence(log_dir, max_iter=86)
