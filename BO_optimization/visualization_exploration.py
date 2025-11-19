#!/usr/bin/env python
"""Generate comprehensive visualization for BO experiment"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def visualize_experiment(log_dir, save_path=None):
    """Generate 9-panel visualization"""

    # Load all iteration data
    files = sorted(glob.glob(f'{log_dir}/iter_*.json'))

    iterations = []
    cvars = []
    scores = []
    acq_values = []
    params_list = []

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            iterations.append(data['iteration'])
            cvars.append(data['cvar'])
            scores.append(data['score'])
            acq_values.append(data['acq_value'])
            params_list.append(data['parameters'])

    cvars = np.array(cvars)
    scores = np.array(scores)
    acq_values = np.array(acq_values)
    iterations = np.array(iterations)

    # Try to load initial sampling data from results JSON
    run_name = os.path.basename(log_dir)
    result_files = glob.glob(f'results/bo_cvar_{run_name.replace("run_", "")}.json')

    initial_cvars = None
    n_initial = 0
    if result_files:
        with open(result_files[0]) as fp:
            result_data = json.load(fp)
            if 'history' in result_data and 'n_initial' in result_data:
                n_initial = result_data['n_initial']
                full_history = result_data['history']
                initial_cvars = np.array(full_history[:n_initial])

    # Find best
    best_idx = np.argmax(cvars)
    best_iter = iterations[best_idx]
    best_cvar = cvars[best_idx]
    best_params = params_list[best_idx]

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. CVaR Progress (with initial sampling if available)
    ax1 = fig.add_subplot(gs[0, 0])

    if initial_cvars is not None and len(initial_cvars) > 0:
        # Plot initial sampling with negative x values
        init_iters = np.arange(-n_initial + 1, 1)
        ax1.plot(init_iters, initial_cvars, 'gray', alpha=0.5, linewidth=2, label='Initial (Random)')
        ax1.scatter(init_iters, initial_cvars, color='gray', s=30, alpha=0.5)
        # Vertical line separating initial and BO
        ax1.axvline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
        # Plot BO iterations
        ax1.plot(iterations, cvars, 'b-', alpha=0.6, linewidth=2, label='BO Iterations')
    else:
        ax1.plot(iterations, cvars, 'b-', alpha=0.6, linewidth=2, label='CVaR')

    ax1.axhline(best_cvar, color='r', linestyle='--', linewidth=2, label=f'Best: {best_cvar:.4f}')
    ax1.scatter([best_iter], [best_cvar], color='red', s=200, zorder=5, marker='*')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('CVaR', fontsize=12)
    ax1.set_title('1. CVaR Progress (with Initial Sampling)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Best
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_best = np.maximum.accumulate(cvars)
    ax2.plot(iterations, cumulative_best, 'g-', linewidth=2)
    ax2.fill_between(iterations, cvars[0], cumulative_best, alpha=0.3, color='green')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best CVaR So Far', fontsize=12)
    ax2.set_title('2. Cumulative Best CVaR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Per-iteration Improvement
    ax3 = fig.add_subplot(gs[0, 2])
    improvements = np.diff(cumulative_best)
    improvement_iters = iterations[1:]
    colors = ['green' if imp > 0 else 'gray' for imp in improvements]
    ax3.bar(improvement_iters, improvements, color=colors, alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Improvement', fontsize=12)
    ax3.set_title('3. Per-iteration Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Best Parameters
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    params_text = f"Best Parameters (Iter {best_iter})\n"
    params_text += f"CVaR: {best_cvar:.4f}\n\n"
    params_text += "AirLine Q:\n"
    params_text += f"  edgeThresh1: {best_params['edgeThresh1']:.2f}\n"
    params_text += f"  simThresh1: {best_params['simThresh1']:.3f}\n"
    params_text += f"  pixelRatio1: {best_params['pixelRatio1']:.3f}\n\n"
    params_text += "AirLine QG:\n"
    params_text += f"  edgeThresh2: {best_params['edgeThresh2']:.2f}\n"
    params_text += f"  simThresh2: {best_params['simThresh2']:.3f}\n"
    params_text += f"  pixelRatio2: {best_params['pixelRatio2']:.3f}\n\n"
    params_text += "RANSAC:\n"
    params_text += f"  weight_q: {best_params['ransac_weight_q']:.2f}\n"
    params_text += f"  weight_qg: {best_params['ransac_weight_qg']:.2f}\n"
    ax4.text(0.1, 0.9, params_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('4. Best Parameters', fontsize=14, fontweight='bold')

    # 5. Statistics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    total_improvement = best_cvar - cvars[0]
    improvement_pct = (total_improvement / cvars[0]) * 100

    n_improvements = np.sum(improvements > 0)
    stats_text = f"Optimization Statistics\n\n"
    stats_text += f"Total iterations: {len(iterations)}\n"
    stats_text += f"Initial CVaR: {cvars[0]:.4f}\n"
    stats_text += f"Final CVaR: {cvars[-1]:.4f}\n"
    stats_text += f"Best CVaR: {best_cvar:.4f}\n"
    stats_text += f"Worst CVaR: {cvars.min():.4f}\n\n"
    stats_text += f"Total improvement: {total_improvement:.4f}\n"
    stats_text += f"Improvement %: {improvement_pct:.1f}%\n"
    stats_text += f"# Improvements: {n_improvements}\n"
    stats_text += f"# Stagnations: {len(improvements) - n_improvements}\n\n"
    stats_text += f"Mean CVaR: {cvars.mean():.4f}\n"
    stats_text += f"Std CVaR: {cvars.std():.4f}\n"

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax5.set_title('5. Statistics', fontsize=14, fontweight='bold')

    # 6. CVaR Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(cvars, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax6.axvline(best_cvar, color='red', linestyle='--', linewidth=2, label=f'Best: {best_cvar:.4f}')
    ax6.axvline(cvars.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {cvars.mean():.4f}')
    ax6.set_xlabel('CVaR', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('6. CVaR Distribution', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Smoothed Trend
    ax7 = fig.add_subplot(gs[2, 0])
    window = min(10, len(cvars) // 5)
    if window > 1:
        smoothed = np.convolve(cvars, np.ones(window)/window, mode='valid')
        smooth_iters = iterations[:len(smoothed)]
        ax7.plot(iterations, cvars, 'b-', alpha=0.3, label='Raw')
        ax7.plot(smooth_iters, smoothed, 'r-', linewidth=2, label=f'MA({window})')
    else:
        ax7.plot(iterations, cvars, 'b-', linewidth=2)
    ax7.set_xlabel('Iteration', fontsize=12)
    ax7.set_ylabel('CVaR', fontsize=12)
    ax7.set_title('7. Smoothed Trend', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Score vs CVaR
    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(scores, cvars, c=iterations, cmap='viridis',
                          s=50, alpha=0.6, edgecolors='black')
    ax8.scatter(scores[best_idx], cvars[best_idx], color='red', s=300,
                marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax8.set_xlabel('Score (Single Evaluation)', fontsize=12)
    ax8.set_ylabel('CVaR (GP Prediction)', fontsize=12)
    ax8.set_title('8. Score vs CVaR', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax8, label='Iteration')
    ax8.grid(True, alpha=0.3)

    # 9. Acquisition Value
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(iterations, acq_values, 'purple', linewidth=2, alpha=0.7)
    ax9.set_xlabel('Iteration', fontsize=12)
    ax9.set_ylabel('Acquisition Value (KG)', fontsize=12)
    ax9.set_title('9. Acquisition Function Value', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(f'BO Experiment Analysis - {len(iterations)} Iterations\n'
                 f'Best CVaR: {best_cvar:.4f} (Iter {best_iter}) | '
                 f'Improvement: {improvement_pct:.1f}%',
                 fontsize=16, fontweight='bold')

    # Save
    if save_path is None:
        save_path = f'results/visualization_exploration_{os.path.basename(log_dir)}.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print(f"EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total iterations: {len(iterations)}")
    print(f"Best CVaR: {best_cvar:.4f} at Iter {best_iter}")
    print(f"Initial CVaR: {cvars[0]:.4f}")
    print(f"Final CVaR: {cvars[-1]:.4f}")
    print(f"Improvement: {total_improvement:.4f} ({improvement_pct:.1f}%)")
    print(f"Mean CVaR: {cvars.mean():.4f} ± {cvars.std():.4f}")
    print(f"Range: [{cvars.min():.4f}, {cvars.max():.4f}]")
    print("="*60)

def visualize_experiment_bo_only(log_dir, save_path=None):
    """Generate visualization with BO iterations only (no initial sampling)"""

    # Load all iteration data
    files = sorted(glob.glob(f'{log_dir}/iter_*.json'))

    iterations = []
    cvars = []
    scores = []
    acq_values = []
    params_list = []

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            iterations.append(data['iteration'])
            cvars.append(data['cvar'])
            scores.append(data['score'])
            acq_values.append(data['acq_value'])
            params_list.append(data['parameters'])

    cvars = np.array(cvars)
    scores = np.array(scores)
    acq_values = np.array(acq_values)
    iterations = np.array(iterations)

    # Find best
    best_idx = np.argmax(cvars)
    best_iter = iterations[best_idx]
    best_cvar = cvars[best_idx]
    best_params = params_list[best_idx]

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. CVaR Progress (BO only)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, cvars, 'b-', alpha=0.6, linewidth=2, label='CVaR')
    ax1.axhline(best_cvar, color='r', linestyle='--', linewidth=2, label=f'Best: {best_cvar:.4f}')
    ax1.scatter([best_iter], [best_cvar], color='red', s=200, zorder=5, marker='*')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('CVaR', fontsize=12)
    ax1.set_title('1. CVaR Progress (BO Iterations Only)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Best
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_best = np.maximum.accumulate(cvars)
    ax2.plot(iterations, cumulative_best, 'g-', linewidth=2)
    ax2.fill_between(iterations, cvars[0], cumulative_best, alpha=0.3, color='green')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best CVaR So Far', fontsize=12)
    ax2.set_title('2. Cumulative Best CVaR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Per-iteration Improvement
    ax3 = fig.add_subplot(gs[0, 2])
    improvements = np.diff(cumulative_best)
    improvement_iters = iterations[1:]
    colors = ['green' if imp > 0 else 'gray' for imp in improvements]
    ax3.bar(improvement_iters, improvements, color=colors, alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Improvement', fontsize=12)
    ax3.set_title('3. Per-iteration Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Best Parameters
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    params_text = f"Best Parameters (Iter {best_iter})\n"
    params_text += f"CVaR: {best_cvar:.4f}\n\n"
    params_text += "AirLine Q:\n"
    params_text += f"  edgeThresh1: {best_params['edgeThresh1']:.2f}\n"
    params_text += f"  simThresh1: {best_params['simThresh1']:.3f}\n"
    params_text += f"  pixelRatio1: {best_params['pixelRatio1']:.3f}\n\n"
    params_text += "AirLine QG:\n"
    params_text += f"  edgeThresh2: {best_params['edgeThresh2']:.2f}\n"
    params_text += f"  simThresh2: {best_params['simThresh2']:.3f}\n"
    params_text += f"  pixelRatio2: {best_params['pixelRatio2']:.3f}\n\n"
    params_text += "RANSAC:\n"
    params_text += f"  weight_q: {best_params['ransac_weight_q']:.2f}\n"
    params_text += f"  weight_qg: {best_params['ransac_weight_qg']:.2f}\n"
    ax4.text(0.1, 0.9, params_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('4. Best Parameters', fontsize=14, fontweight='bold')

    # 5. Statistics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    total_improvement = best_cvar - cvars[0]
    improvement_pct = (total_improvement / cvars[0]) * 100

    n_improvements = np.sum(improvements > 0)
    stats_text = f"Optimization Statistics\n\n"
    stats_text += f"Total iterations: {len(iterations)}\n"
    stats_text += f"Initial CVaR: {cvars[0]:.4f}\n"
    stats_text += f"Final CVaR: {cvars[-1]:.4f}\n"
    stats_text += f"Best CVaR: {best_cvar:.4f}\n"
    stats_text += f"Worst CVaR: {cvars.min():.4f}\n\n"
    stats_text += f"Total improvement: {total_improvement:.4f}\n"
    stats_text += f"Improvement %: {improvement_pct:.1f}%\n"
    stats_text += f"# Improvements: {n_improvements}\n"
    stats_text += f"# Stagnations: {len(improvements) - n_improvements}\n\n"
    stats_text += f"Mean CVaR: {cvars.mean():.4f}\n"
    stats_text += f"Std CVaR: {cvars.std():.4f}\n"

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax5.set_title('5. Statistics', fontsize=14, fontweight='bold')

    # 6. CVaR Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(cvars, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax6.axvline(best_cvar, color='red', linestyle='--', linewidth=2, label=f'Best: {best_cvar:.4f}')
    ax6.axvline(cvars.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {cvars.mean():.4f}')
    ax6.set_xlabel('CVaR', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('6. CVaR Distribution', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Smoothed Trend
    ax7 = fig.add_subplot(gs[2, 0])
    window = min(10, len(cvars) // 5)
    if window > 1:
        smoothed = np.convolve(cvars, np.ones(window)/window, mode='valid')
        smooth_iters = iterations[:len(smoothed)]
        ax7.plot(iterations, cvars, 'b-', alpha=0.3, label='Raw')
        ax7.plot(smooth_iters, smoothed, 'r-', linewidth=2, label=f'MA({window})')
    else:
        ax7.plot(iterations, cvars, 'b-', linewidth=2)
    ax7.set_xlabel('Iteration', fontsize=12)
    ax7.set_ylabel('CVaR', fontsize=12)
    ax7.set_title('7. Smoothed Trend', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Score vs CVaR
    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(scores, cvars, c=iterations, cmap='viridis',
                          s=50, alpha=0.6, edgecolors='black')
    ax8.scatter(scores[best_idx], cvars[best_idx], color='red', s=300,
                marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax8.set_xlabel('Score (Single Evaluation)', fontsize=12)
    ax8.set_ylabel('CVaR (GP Prediction)', fontsize=12)
    ax8.set_title('8. Score vs CVaR', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax8, label='Iteration')
    ax8.grid(True, alpha=0.3)

    # 9. Acquisition Value
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(iterations, acq_values, 'purple', linewidth=2, alpha=0.7)
    ax9.set_xlabel('Iteration', fontsize=12)
    ax9.set_ylabel('Acquisition Value (KG)', fontsize=12)
    ax9.set_title('9. Acquisition Function Value', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(f'BO Experiment Analysis - {len(iterations)} Iterations\n'
                 f'Best CVaR: {best_cvar:.4f} (Iter {best_iter}) | '
                 f'Improvement: {improvement_pct:.1f}%',
                 fontsize=16, fontweight='bold')

    # Save
    if save_path is None:
        save_path = f'results/visualization_bo_only_{os.path.basename(log_dir)}.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "logs/run_20251117_111151"

    # Generate both versions
    print("Generating visualization WITH initial sampling...")
    visualize_experiment(log_dir)

    print("\nGenerating visualization WITHOUT initial sampling (BO only)...")
    visualize_experiment_bo_only(log_dir)
