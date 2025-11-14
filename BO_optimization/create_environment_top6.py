"""
Top 6 특징만 추출해서 새로운 environment JSON 생성
- Input: environment_roi_worst_case_v2.json (15D)
- Output: environment_top6.json (6D)
"""

import json

# Top 6 features (상관관계 기준 선택)
TOP6_FEATURES = [
    'local_contrast',      # r = -0.42 (STRONG)
    'clip_rough',          # r =  0.40 (STRONG)
    'brightness',          # r =  0.22 (MODERATE)
    'clip_smooth',         # r =  0.21 (MODERATE)
    'gradient_strength',   # r = -0.21 (MODERATE)
    'edge_density'         # r =  0.20 (WEAK)
]


def create_top6_environment(input_file, output_file):
    """
    15D 환경 파일에서 Top 6 특징만 추출

    Args:
        input_file: 15D environment JSON (worst_case_v2)
        output_file: 6D environment JSON
    """
    print("="*70)
    print("Creating Top 6 Feature Environment File")
    print("="*70)

    # 1. Load 15D environment
    print(f"\n[1] Loading {input_file}...")
    with open(input_file) as f:
        env_15d = json.load(f)

    print(f"  Loaded {len(env_15d)} images")

    # 2. Extract top 6 features
    print(f"\n[2] Extracting top 6 features...")
    print(f"  Selected features:")
    for i, feat in enumerate(TOP6_FEATURES, 1):
        print(f"    {i}. {feat}")

    env_6d = {}
    missing_features = []

    for img_name, features in env_15d.items():
        env_6d[img_name] = {}

        for feat in TOP6_FEATURES:
            if feat in features:
                env_6d[img_name][feat] = features[feat]
            else:
                if feat not in missing_features:
                    missing_features.append(feat)
                    print(f"  [WARN] Feature '{feat}' not found in {img_name}")

    # 3. Verify
    print(f"\n[3] Verification...")
    sample_img = list(env_6d.keys())[0]
    sample_features = env_6d[sample_img]

    print(f"  Sample image: {sample_img}")
    print(f"  Features (6D):")
    for feat, val in sample_features.items():
        print(f"    {feat:<20}: {val:.6f}")

    # Check all images have 6 features
    all_6d = all(len(features) == 6 for features in env_6d.values())
    if all_6d:
        print(f"\n  [OK] All {len(env_6d)} images have 6D features")
    else:
        counts = {}
        for features in env_6d.values():
            dim = len(features)
            counts[dim] = counts.get(dim, 0) + 1
        print(f"\n  [WARN] Not all images have 6D:")
        for dim, count in sorted(counts.items()):
            print(f"    {dim}D: {count} images")

    # 4. Save
    print(f"\n[4] Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(env_6d, f, indent=2)

    print(f"\n[OK] Done! Created {output_file} with 6D features")

    # 5. Statistics
    print(f"\n[5] Feature Statistics:")
    import numpy as np

    for feat in TOP6_FEATURES:
        values = [env_6d[img][feat] for img in env_6d if feat in env_6d[img]]
        if values:
            print(f"  {feat:<20}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                  f"min={np.min(values):.4f}, max={np.max(values):.4f}")

    print("\n" + "="*70)
    print("Next step:")
    print("  Update optimization.py to use 'environment_top6.json'")
    print("  Set w_dim = 6")
    print("  Run BoRisk optimization!")
    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract top 6 features")
    parser.add_argument("--input", default="environment_roi_worst_case_v2.json",
                       help="Input 15D environment file")
    parser.add_argument("--output", default="environment_top6.json",
                       help="Output 6D environment file")
    args = parser.parse_args()

    create_top6_environment(args.input, args.output)


if __name__ == "__main__":
    main()
