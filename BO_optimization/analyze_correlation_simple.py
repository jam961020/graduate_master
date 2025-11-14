"""
Simple correlation analysis between environment features and performance
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_correlation(log_dir, env_file, gt_file="../dataset/ground_truth.json"):
    """
    Analyze correlation between environment features and performance

    Args:
        log_dir: Experiment log directory
        env_file: Environment features JSON file
        gt_file: Ground truth JSON file
    """
    print("="*70)
    print("Environment Feature-Performance Correlation Analysis")
    print("="*70)

    # 1. Load experiment logs
    print("\n[1] Loading experiment logs...")
    log_dir = Path(log_dir)
    image_scores = defaultdict(list)

    for log_file in sorted(log_dir.glob("iter_*.json")):
        with open(log_file) as f:
            data = json.load(f)
            image_scores[data['image_idx']].append(data['score'])

    # Calculate mean performance per image
    image_perf = {}
    for img_idx, scores in image_scores.items():
        image_perf[img_idx] = np.mean(scores)

    print(f"  Loaded {len(image_perf)} evaluated images")
    print(f"  Performance range: [{min(image_perf.values()):.4f}, {max(image_perf.values()):.4f}]")

    # 2. Load GT to get image names
    print("\n[2] Loading ground truth...")
    with open(gt_file) as f:
        gt_data = json.load(f)
    image_names = list(gt_data.keys())
    print(f"  Total GT images: {len(image_names)}")

    # 3. Load environment features
    print("\n[3] Loading environment features...")
    with open(env_file) as f:
        env_features = json.load(f)

    print(f"  Loaded features for {len(env_features)} images")

    # Get feature names from first image
    sample_features = list(env_features.values())[0]
    feature_names = list(sample_features.keys())
    print(f"  Features: {', '.join(feature_names)}")

    # 4. Match images
    print("\n[4] Matching images...")
    matched_data = []

    for img_idx, perf in image_perf.items():
        if img_idx < len(image_names):
            img_name = image_names[img_idx]
            if img_name in env_features:
                matched_data.append({
                    'idx': img_idx,
                    'name': img_name,
                    'perf': perf,
                    'features': env_features[img_name]
                })

    print(f"  Matched {len(matched_data)} images")

    if len(matched_data) < 5:
        print("\n[ERROR] Too few matched images for correlation analysis!")
        return None

    # 5. Correlation analysis
    print("\n[5] Correlation Analysis")
    print("="*70)

    print(f"\n{'Feature':<25} {'Correlation':<12} {'Strength':<15} {'Interpretation'}")
    print("-" * 80)

    correlations = []
    for fname in feature_names:
        feature_vals = [d['features'][fname] for d in matched_data]
        perf_vals = [d['perf'] for d in matched_data]

        # Check for constant features
        if np.std(feature_vals) < 1e-10:
            print(f"{fname:<25} {'N/A':<12}    CONSTANT (std=0)")
            continue

        corr = np.corrcoef(feature_vals, perf_vals)[0, 1]
        correlations.append((fname, corr))

        # Determine strength
        abs_corr = abs(corr)
        if abs_corr > 0.5:
            strength = "VERY STRONG"
            marker = "***"
        elif abs_corr > 0.3:
            strength = "STRONG"
            marker = "** "
        elif abs_corr > 0.2:
            strength = "MODERATE"
            marker = "*  "
        elif abs_corr > 0.1:
            strength = "WEAK"
            marker = "   "
        else:
            strength = "NEGLIGIBLE"
            marker = "   "

        # Interpretation
        if corr > 0:
            interp = "Higher = Better"
        else:
            interp = "Lower = Better"

        print(f"{fname:<25} {corr:>10.4f} {marker} {strength:<15} {interp}")

    # 6. Summary
    print(f"\n[6] Summary")
    print("="*70)

    if not correlations:
        print("\n[ERROR] No valid correlations computed!")
        return None

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 5 features by absolute correlation:")
    for i, (fname, corr) in enumerate(correlations[:5], 1):
        print(f"  {i}. {fname:<25}: r = {corr:>7.4f}")

    print(f"\nBest overall: {correlations[0][0]} = {correlations[0][1]:.4f}")

    # Statistics
    abs_corrs = [abs(c) for _, c in correlations]
    print(f"\nCorrelation statistics:")
    print(f"  Mean |r|: {np.mean(abs_corrs):.4f}")
    print(f"  Max  |r|: {np.max(abs_corrs):.4f}")
    print(f"  Min  |r|: {np.min(abs_corrs):.4f}")

    # Strong features
    strong_features = [(f, c) for f, c in correlations if abs(c) > 0.2]
    if strong_features:
        print(f"\n[OK] {len(strong_features)} features with |r| > 0.2:")
        for fname, corr in strong_features:
            print(f"  - {fname}: {corr:.3f}")
    else:
        print(f"\n[WARN] No features with |r| > 0.2")
        print(f"  Environment features may not be predictive of performance")

    return correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="Experiment log directory")
    parser.add_argument("--env_file", required=True, help="Environment features JSON file")
    parser.add_argument("--gt_file", default="../dataset/ground_truth.json", help="Ground truth JSON")
    args = parser.parse_args()

    correlations = analyze_correlation(args.log_dir, args.env_file, args.gt_file)

    if correlations:
        print("\n[SUCCESS] Correlation analysis completed!")
    else:
        print("\n[FAILED] Correlation analysis failed!")


if __name__ == "__main__":
    main()
