"""
다양한 환경 조합의 상관관계 분석
Baseline만, CLIP만, 조합 등 비교
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_environment_combinations(log_dir, env_file):
    """
    다양한 환경 조합별 상관관계 분석
    """
    print("="*70)
    print("Environment Combination Analysis")
    print("="*70)

    # 1. Load experiment logs
    print("\n[1] Loading experiment logs...")
    log_dir = Path(log_dir)
    image_scores = defaultdict(list)

    for log_file in sorted(log_dir.glob("iter_*.json")):
        with open(log_file) as f:
            data = json.load(f)
            image_scores[data['image_idx']].append(data['score'])

    image_perf = {}
    for img_idx, scores in image_scores.items():
        image_perf[img_idx] = np.mean(scores)

    print(f"  Loaded {len(image_perf)} evaluated images")

    # 2. Load GT
    with open("../dataset/ground_truth.json") as f:
        gt_data = json.load(f)
    image_names = list(gt_data.keys())

    # 3. Load environment features
    print("\n[2] Loading environment features...")
    with open(env_file) as f:
        env_features = json.load(f)

    # 4. Match images
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

    # 5. Feature grouping
    sample_features = matched_data[0]['features']
    all_features = list(sample_features.keys())

    baseline_features = [f for f in all_features if not f.startswith('clip_')]
    clip_features = [f for f in all_features if f.startswith('clip_')]

    print(f"\n  Baseline: {len(baseline_features)} features")
    print(f"  CLIP: {len(clip_features)} features")

    # 6. Analyze combinations
    print("\n[3] Analyzing Combinations")
    print("="*70)

    combinations = [
        ("Baseline Only", baseline_features),
        ("CLIP Only", clip_features),
        ("Baseline + CLIP", all_features),
    ]

    results = {}

    for combo_name, feature_list in combinations:
        print(f"\n{combo_name}:")
        print("-" * 70)

        correlations = []
        for fname in feature_list:
            feature_vals = [d['features'][fname] for d in matched_data]
            perf_vals = [d['perf'] for d in matched_data]

            corr = np.corrcoef(feature_vals, perf_vals)[0, 1]
            correlations.append((fname, corr))

        # Sort by absolute value
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        # Top 5
        print(f"\nTop 5 features:")
        for fname, corr in correlations[:5]:
            strength = "MODERATE" if abs(corr) > 0.25 else ("WEAK" if abs(corr) > 0.1 else "NEGLIGIBLE")
            print(f"  {fname:<25}: r = {corr:>7.4f}  ({strength})")

        # Stats
        abs_corrs = [abs(c) for _, c in correlations]
        results[combo_name] = {
            'features': feature_list,
            'correlations': correlations,
            'best': correlations[0] if correlations else ('N/A', 0.0),
            'mean_abs_corr': np.mean(abs_corrs) if abs_corrs else 0.0,
            'num_moderate': sum(1 for c in abs_corrs if c > 0.25),
            'num_weak': sum(1 for c in abs_corrs if 0.1 < c <= 0.25),
        }

    # 7. Comparison
    print(f"\n[4] Summary")
    print("="*70)

    print(f"\n{'Combination':<20} {'Best |r|':<12} {'Mean |r|':<12} {'>0.25':<8} {'0.1-0.25':<10}")
    print("-" * 70)

    for combo_name, result in results.items():
        best_corr = abs(result['best'][1])
        mean_corr = result['mean_abs_corr']
        num_moderate = result['num_moderate']
        num_weak = result['num_weak']

        print(f"{combo_name:<20} {best_corr:<12.4f} {mean_corr:<12.4f} {num_moderate:<8} {num_weak:<10}")

    # 8. Recommendation
    print(f"\n[5] Recommendation")
    print("="*70)

    best_combo = max(results.items(), key=lambda x: abs(x[1]['best'][1]))
    best_name, best_result = best_combo

    print(f"\nBest combination: {best_name}")
    print(f"  Top feature: {best_result['best'][0]} = {best_result['best'][1]:.4f}")
    print(f"  Features with |r| > 0.25: {best_result['num_moderate']}")

    if best_result['num_moderate'] >= 3:
        print(f"\n✅ Use {best_name} for BoRisk!")
    elif best_result['num_moderate'] >= 1:
        print(f"\n⚠️  {best_name} is usable but consider adding more features")
    else:
        print(f"\n❌ Correlation too weak - consider parameter-only optimization")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/run_20251113_225648")
    parser.add_argument("--env_file", default="environment_roi_v2.json")
    args = parser.parse_args()

    results = analyze_environment_combinations(args.log_dir, args.env_file)


if __name__ == "__main__":
    main()
