"""
Environment-specific performance analysis.
Analyzes validation results by environmental features (brightness, edge density, blur).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_validation_results(val_results_file):
    """Load validation results JSON."""
    with open(val_results_file, 'r') as f:
        return json.load(f)


def group_by_feature(scores, features, feature_idx, n_groups=3):
    """
    Group scores by a specific environmental feature.

    Args:
        scores: Array of scores
        features: Array of environmental features (N x 6)
        feature_idx: Index of feature to group by (0-5)
        n_groups: Number of groups (default: 3)

    Returns:
        dict: Grouped scores and statistics
    """
    feature_values = features[:, feature_idx]

    # Calculate quantiles for grouping
    quantiles = np.linspace(0, 1, n_groups + 1)
    thresholds = np.quantile(feature_values, quantiles)

    groups = []
    for i in range(n_groups):
        if i == n_groups - 1:
            mask = (feature_values >= thresholds[i]) & (feature_values <= thresholds[i+1])
        else:
            mask = (feature_values >= thresholds[i]) & (feature_values < thresholds[i+1])

        group_scores = scores[mask]
        groups.append({
            'range': (thresholds[i], thresholds[i+1]),
            'scores': group_scores,
            'mean': np.mean(group_scores),
            'std': np.std(group_scores),
            'cvar': np.mean(np.sort(group_scores)[:max(1, int(0.3 * len(group_scores)))]),
            'count': len(group_scores)
        })

    return groups


def analyze_by_environments(val_results_file, env_features_file, alpha=0.3):
    """
    Analyze performance by environmental features.

    Args:
        val_results_file: Path to validation results JSON
        env_features_file: Path to environment features JSON
        alpha: CVaR threshold
    """
    # Load validation results
    val_results = load_validation_results(val_results_file)
    scores = np.array(val_results['scores'])

    # Load environment features
    with open(env_features_file, 'r') as f:
        env_data = json.load(f)

    # Extract features for validation images
    # Assume validation_images.json order matches scores order
    with open('validation_images.json', 'r') as f:
        val_names = json.load(f)

    # Match environment features to validation images
    features = []
    matched_scores = []

    for i, img_name in enumerate(val_names):
        if img_name in env_data:
            features.append(env_data[img_name])
            matched_scores.append(scores[i])
        else:
            print(f"Warning: No environment features for {img_name}")

    features = np.array(features)
    matched_scores = np.array(matched_scores)

    print(f"Matched {len(matched_scores)}/{len(scores)} images with environment features")

    # Feature names
    feature_names = [
        'Brightness',
        'Contrast',
        'Edge Density',
        'Texture Complexity',
        'Blur Level',
        'Noise Level'
    ]

    # Analyze by each feature
    results = {}

    for idx, name in enumerate(feature_names):
        print(f"\nAnalyzing by {name}...")
        groups = group_by_feature(matched_scores, features, idx, n_groups=3)

        results[name] = groups

        # Print summary
        group_labels = ['Low', 'Medium', 'High']
        print(f"\n{name} Analysis:")
        print("-" * 60)
        for i, (label, group) in enumerate(zip(group_labels, groups)):
            print(f"{label:10s} [{group['range'][0]:.3f}, {group['range'][1]:.3f}]: "
                  f"Mean={group['mean']:.4f}, CVaR={group['cvar']:.4f}, N={group['count']}")

    return results, feature_names


def plot_environment_analysis(results, feature_names, output_file='environment_analysis.png'):
    """
    Create visualization of environment-specific performance.

    Args:
        results: Dictionary of analysis results
        feature_names: List of feature names
        output_file: Output image file
    """
    # Select top 3 most interesting features for paper
    selected_features = ['Brightness', 'Edge Density', 'Blur Level']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    group_labels = ['Low', 'Medium', 'High']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for ax, feature_name in zip(axes, selected_features):
        if feature_name not in results:
            continue

        groups = results[feature_name]

        # Extract data
        means = [g['mean'] for g in groups]
        cvars = [g['cvar'] for g in groups]
        stds = [g['std'] for g in groups]

        x = np.arange(len(group_labels))
        width = 0.35

        # Plot bars
        bars1 = ax.bar(x - width/2, means, width, label='Mean Score',
                       color=colors[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, cvars, width, label='CVaR',
                       color=colors[2], alpha=0.8)

        # Add error bars for mean
        ax.errorbar(x - width/2, means, yerr=stds, fmt='none',
                    ecolor='black', capsize=3, alpha=0.5)

        # Styling
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Performance by {feature_name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")


def create_table(results, feature_names):
    """Create LaTeX table for paper."""
    selected_features = ['Brightness', 'Edge Density', 'Blur Level']
    group_labels = ['Low', 'Medium', 'High']

    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Performance Analysis by Environmental Features}")
    print(r"\label{tab:env_performance}")
    print(r"\begin{tabular}{l|ccc|ccc}")
    print(r"\hline")
    print(r"Feature & \multicolumn{3}{c|}{Mean Score} & \multicolumn{3}{c}{CVaR ($\alpha$=0.3)} \\")
    print(r"        & Low & Medium & High & Low & Medium & High \\")
    print(r"\hline")

    for feature_name in selected_features:
        if feature_name not in results:
            continue

        groups = results[feature_name]
        means = [g['mean'] for g in groups]
        cvars = [g['cvar'] for g in groups]

        print(f"{feature_name:15s} & "
              f"{means[0]:.3f} & {means[1]:.3f} & {means[2]:.3f} & "
              f"{cvars[0]:.3f} & {cvars[1]:.3f} & {cvars[2]:.3f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze environment-specific performance')
    parser.add_argument('--val_results', type=str, default='validation_results.json',
                        help='Validation results JSON')
    parser.add_argument('--env_features', type=str, default='environment_top6.json',
                        help='Environment features JSON')
    parser.add_argument('--output_plot', type=str, default='results/environment_analysis.png',
                        help='Output plot file')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='CVaR alpha')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.val_results).exists():
        print(f"Error: {args.val_results} not found")
        print("Please run validate_best_params.py first!")
        return

    if not Path(args.env_features).exists():
        print(f"Error: {args.env_features} not found")
        return

    # Analyze
    results, feature_names = analyze_by_environments(
        args.val_results,
        args.env_features,
        alpha=args.alpha
    )

    # Plot
    plot_environment_analysis(results, feature_names, args.output_plot)

    # Create table
    create_table(results, feature_names)


if __name__ == '__main__':
    main()
