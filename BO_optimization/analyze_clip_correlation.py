"""
Analyze correlation between CLIP features and performance
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_clip_correlation(log_dir, clip_features_file):
    """
    Analyze correlation between CLIP environment features and performance
    
    Args:
        log_dir: Experiment log directory
        clip_features_file: CLIP features JSON file
    """
    print("="*70)
    print("CLIP Feature-Performance Correlation Analysis")
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
    
    # 2. Load GT to get image names
    print("\n[2] Loading ground truth...")
    with open("../dataset/ground_truth.json") as f:
        gt_data = json.load(f)
    image_names = list(gt_data.keys())
    
    # 3. Load CLIP features
    print("\n[3] Loading CLIP features...")
    with open(clip_features_file) as f:
        clip_features = json.load(f)
    
    print(f"  Loaded CLIP features for {len(clip_features)} images")
    
    # 4. Match images
    print("\n[4] Matching images...")
    matched_data = []
    
    for img_idx, perf in image_perf.items():
        if img_idx < len(image_names):
            img_name = image_names[img_idx]
            if img_name in clip_features:
                matched_data.append({
                    'idx': img_idx,
                    'name': img_name,
                    'perf': perf,
                    'features': clip_features[img_name]
                })
    
    print(f"  Matched {len(matched_data)} images")
    
    # 5. Correlation analysis
    print("\n[5] Correlation Analysis")
    print("="*70)
    
    feature_names = ['clip_clear', 'clip_shadow', 'clip_debris', 
                     'clip_reflection', 'clip_beads', 'clip_noise']
    
    print(f"\n{'Feature':<20} {'Correlation':<12} {'Strength':<15} {'Interpretation'}")
    print("-" * 75)
    
    correlations = []
    for fname in feature_names:
        feature_vals = [d['features'][fname] for d in matched_data]
        perf_vals = [d['perf'] for d in matched_data]
        
        corr = np.corrcoef(feature_vals, perf_vals)[0, 1]
        correlations.append((fname, corr))
        
        if abs(corr) > 0.5:
            strength = "VERY STRONG"
            marker = "***"
        elif abs(corr) > 0.3:
            strength = "STRONG"
            marker = "** "
        elif abs(corr) > 0.2:
            strength = "MODERATE"
            marker = "*  "
        elif abs(corr) > 0.1:
            strength = "WEAK"
            marker = "   "
        else:
            strength = "NEGLIGIBLE"
            marker = "   "
        
        if corr > 0:
            interp = "Higher = Better"
        else:
            interp = "Lower = Better"
        
        print(f"{fname:<20} {corr:>10.4f} {marker} {strength:<15} {interp}")
    
    # 6. Compare with baseline
    print(f"\n[6] Comparison with Baseline")
    print("="*70)
    
    print("\nBaseline (brightness, contrast, etc.):")
    print("  All features: |r| < 0.15")
    print("  Best: contrast r = -0.135")
    
    print(f"\nCLIP features:")
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, corr in correlations[:3]:
        print(f"  {fname}: r = {corr:.3f}")
    
    best_clip_corr = abs(correlations[0][1])
    improvement = (best_clip_corr / 0.135 - 1) * 100 if best_clip_corr > 0 else 0
    
    print(f"\nBest CLIP correlation: {correlations[0][0]} = {correlations[0][1]:.3f}")
    print(f"Improvement vs baseline: {improvement:.1f}%")
    
    # 7. Recommendations
    print(f"\n[7] Recommendations")
    print("="*70)
    
    strong_features = [f for f, c in correlations if abs(c) > 0.25]
    
    if strong_features:
        print(f"\n✅ SUCCESS! CLIP features show stronger correlation!")
        print(f"\nStrong features (|r| > 0.25):")
        for fname in strong_features:
            corr_val = [c for f, c in correlations if f == fname][0]
            print(f"  - {fname}: {corr_val:.3f}")
        
        print(f"\nNext steps:")
        print(f"  1. Use CLIP environment in BoRisk optimization")
        print(f"  2. Run comparison experiment (baseline vs CLIP)")
        print(f"  3. Expect better CVaR convergence")
    
    else:
        print(f"\n⚠️  CLIP features still weak (|r| < 0.25)")
        print(f"\nPossible reasons:")
        print(f"  1. ROI detection quality issues")
        print(f"  2. Need different CLIP prompts")
        print(f"  3. Performance driven by parameters, not environment")
        
        print(f"\nOptions:")
        print(f"  A. Try welding-specific physical features")
        print(f"  B. Simplify to parameter-only optimization")
        print(f"  C. Hybrid: CLIP + physical features")
    
    return correlations


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/run_20251113_225648",
                       help="Experiment log directory")
    parser.add_argument("--clip_features", default="environment_clip.json",
                       help="CLIP features JSON file")
    args = parser.parse_args()
    
    correlations = analyze_clip_correlation(args.log_dir, args.clip_features)


if __name__ == "__main__":
    main()
