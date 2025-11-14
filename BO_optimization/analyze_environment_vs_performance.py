"""
환경 특징 vs CVaR/Score 상관관계 분석
KG가 양수인데 CVaR이 개선 안 되는 이유 찾기
"""

import json
import numpy as np
from pathlib import Path
import argparse


def analyze_env_vs_performance(log_dir, env_file, gt_file="../dataset/ground_truth.json"):
    """
    환경 특징과 CVaR/Score의 상관관계 분석
    """
    print("="*80)
    print("Environment Features vs CVaR/Score Correlation Analysis")
    print("="*80)

    log_dir = Path(log_dir)

    # 1. Load experiment logs
    print("\n[1] Loading experiment logs...")
    data = []
    for log_file in sorted(log_dir.glob("iter_*.json")):
        with open(log_file) as f:
            d = json.load(f)
            data.append({
                'iter': d['iteration'],
                'image_idx': d['image_idx'],
                'w_idx': d.get('w_idx', 0),
                'cvar': d['cvar'],
                'score': d['score'],
                'acq_value': d.get('acq_value', 0)
            })

    print(f"  Loaded {len(data)} iterations")

    # 2. Load GT to get image names
    print("\n[2] Loading ground truth...")
    with open(gt_file) as f:
        gt_data = json.load(f)
    image_names = list(gt_data.keys())

    # 3. Load environment features
    print("\n[3] Loading environment features...")
    with open(env_file) as f:
        env_features = json.load(f)

    sample_features = list(env_features.values())[0]
    feature_names = list(sample_features.keys())
    print(f"  Features: {', '.join(feature_names)}")

    # 4. Match data with environment features
    print("\n[4] Matching data with environment features...")
    matched_data = []
    for d in data:
        if d['image_idx'] < len(image_names):
            img_name = image_names[d['image_idx']]
            if img_name in env_features:
                matched_data.append({
                    **d,
                    'image_name': img_name,
                    'env': env_features[img_name]
                })

    print(f"  Matched {len(matched_data)} iterations")

    # 5. Correlation analysis
    print("\n[5] Correlation Analysis")
    print("="*80)

    # Extract arrays
    cvars = np.array([d['cvar'] for d in matched_data])
    scores = np.array([d['score'] for d in matched_data])
    acq_values = np.array([d['acq_value'] for d in matched_data])

    print(f"\nCVaR statistics:")
    print(f"  Mean: {cvars.mean():.4f}")
    print(f"  Std:  {cvars.std():.4f}")
    print(f"  Min:  {cvars.min():.4f}")
    print(f"  Max:  {cvars.max():.4f}")

    print(f"\nScore statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")

    print(f"\nAcq Value statistics:")
    print(f"  Mean: {acq_values.mean():.4f}")
    print(f"  Std:  {acq_values.std():.4f}")
    print(f"  Min:  {acq_values.min():.4f}")
    print(f"  Max:  {acq_values.max():.4f}")

    # CVaR vs Score correlation
    cvar_score_corr = np.corrcoef(cvars, scores)[0, 1]
    print(f"\n*** CVaR vs Score correlation: {cvar_score_corr:.4f} ***")

    # KG (acq_value) vs actual CVaR improvement
    cvar_improvements = np.diff(cvars, prepend=cvars[0])
    if len(acq_values) == len(cvar_improvements):
        kg_vs_improvement = np.corrcoef(acq_values, cvar_improvements)[0, 1]
        print(f"*** KG prediction vs Actual CVaR improvement: {kg_vs_improvement:.4f} ***")

    # 6. Environment features vs CVaR
    print("\n[6] Environment Features vs CVaR")
    print("="*80)
    print(f"\n{'Feature':<25} {'vs CVaR':<12} {'vs Score':<12} {'Strength'}")
    print("-" * 80)

    results = []
    for fname in feature_names:
        feature_vals = np.array([d['env'][fname] for d in matched_data])

        # Check for constant features
        if np.std(feature_vals) < 1e-10:
            print(f"{fname:<25} {'N/A':<12} {'N/A':<12} CONSTANT")
            continue

        # Correlation with CVaR
        corr_cvar = np.corrcoef(feature_vals, cvars)[0, 1]

        # Correlation with Score
        corr_score = np.corrcoef(feature_vals, scores)[0, 1]

        results.append((fname, corr_cvar, corr_score))

        # Strength
        abs_max = max(abs(corr_cvar), abs(corr_score))
        if abs_max > 0.5:
            strength = "VERY STRONG"
        elif abs_max > 0.3:
            strength = "STRONG"
        elif abs_max > 0.2:
            strength = "MODERATE"
        elif abs_max > 0.1:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"

        print(f"{fname:<25} {corr_cvar:>10.4f}   {corr_score:>10.4f}   {strength}")

    # 7. Summary
    print(f"\n[7] Summary")
    print("="*80)

    # Sort by CVaR correlation
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 5 features by |correlation with CVaR|:")
    for i, (fname, corr_cvar, corr_score) in enumerate(results[:5], 1):
        print(f"  {i}. {fname:<25}: CVaR r={corr_cvar:>7.4f}, Score r={corr_score:>7.4f}")

    # Sort by Score correlation
    results.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\nTop 5 features by |correlation with Score|:")
    for i, (fname, corr_cvar, corr_score) in enumerate(results[:5], 1):
        print(f"  {i}. {fname:<25}: CVaR r={corr_cvar:>7.4f}, Score r={corr_score:>7.4f}")

    # 8. Diagnosis
    print(f"\n[8] Diagnosis")
    print("="*80)

    if cvar_score_corr < 0.3:
        print(f"\nPROBLEM: CVaR vs Score correlation is LOW ({cvar_score_corr:.4f})")
        print("  -> CVaR estimates are not aligned with actual performance!")
        print("  -> GP model may be predicting poorly across environments")

    # Check if environment features predict CVaR better than Score
    avg_cvar_corr = np.mean([abs(x[1]) for x in results])
    avg_score_corr = np.mean([abs(x[2]) for x in results])

    print(f"\nAverage |correlation|:")
    print(f"  Environment vs CVaR:  {avg_cvar_corr:.4f}")
    print(f"  Environment vs Score: {avg_score_corr:.4f}")

    if avg_score_corr > avg_cvar_corr:
        print(f"\nOBSERVATION: Environment features predict Score better than CVaR")
        print("  -> CVaR calculation may be problematic")
        print("  -> GP predictions across w_set may be inaccurate")

    # KG analysis
    if len(acq_values) > 0:
        positive_kg = np.sum(acq_values > 0)
        print(f"\nKG Analysis:")
        print(f"  Positive KG predictions: {positive_kg}/{len(acq_values)} ({positive_kg/len(acq_values)*100:.1f}%)")
        if positive_kg > len(acq_values) * 0.8:
            print("  -> KG is almost always positive!")
            if cvars[-1] < cvars[0]:
                print("  -> But CVaR DECREASED overall!")
                print("  -> KG predictions are WRONG!")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="Experiment log directory")
    parser.add_argument("--env_file", required=True, help="Environment features JSON")
    parser.add_argument("--gt_file", default="../dataset/ground_truth.json", help="Ground truth JSON")
    args = parser.parse_args()

    results = analyze_env_vs_performance(args.log_dir, args.env_file, args.gt_file)

    print("\n[SUCCESS] Analysis completed!")


if __name__ == "__main__":
    main()
