"""
랜덤 이미지 평가 스크립트
- 통계적 신뢰도 향상을 위해 더 많은 이미지 평가
- Default 파라미터로 평가 (BO 없이)
- 상관관계 재분석용
"""

import json
import numpy as np
import random
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

from full_pipeline import detect_with_full_pipeline
from optimization import line_equation_evaluation
from yolo_detector import YOLODetector


# Default parameters (중간값 사용)
DEFAULT_PARAMS = {
    'edgeThresh1': -3.0,
    'simThresh1': 0.98,
    'pixelRatio1': 0.05,
    'edgeThresh2': 1.0,
    'simThresh2': 0.75,
    'pixelRatio2': 0.05
}


def evaluate_single_image(image, gt_lines, params, yolo_detector):
    """
    단일 이미지 평가

    Args:
        image: cv2로 로드된 이미지
        gt_lines: GT 라인 dict
        params: 파라미터 dict
        yolo_detector: YOLO 검출기

    Returns:
        score: 평가 점수
    """
    # RANSAC weights
    ransac_weights = (2.0, 5.0)  # (Q, QG)

    # Full pipeline 실행
    detected_coords = detect_with_full_pipeline(image, params, yolo_detector, ransac_weights)

    # Evaluation
    h, w = image.shape[:2]
    score = line_equation_evaluation(detected_coords, gt_lines, image_size=(w, h))

    return score


def main():
    parser = argparse.ArgumentParser(description="Evaluate random images")
    parser.add_argument("--n_images", type=int, default=100,
                       help="Number of images to evaluate")
    parser.add_argument("--image_dir", default="../dataset/images/test",
                       help="Image directory")
    parser.add_argument("--gt_file", default="../dataset/ground_truth.json",
                       help="Ground truth file")
    parser.add_argument("--output_dir", default="logs_random",
                       help="Output directory for evaluation logs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    print("="*70)
    print("Random Image Evaluation (for Correlation Analysis)")
    print("="*70)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load ground truth
    print(f"\n[1] Loading ground truth from {args.gt_file}...")
    with open(args.gt_file) as f:
        gt_data = json.load(f)

    print(f"  Total images in GT: {len(gt_data)}")

    # 2. Select random images
    print(f"\n[2] Selecting {args.n_images} random images...")
    image_names = list(gt_data.keys())
    n_images = min(args.n_images, len(image_names))
    selected_images = random.sample(image_names, n_images)

    print(f"  Selected {n_images} images (seed={args.seed})")

    # 3. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\n[3] Output directory: {output_dir}")

    # 4. Initialize YOLO detector
    print(f"\n[4] Initializing YOLO detector...")
    yolo_detector = YOLODetector("models/best.pt")

    # 5. Evaluate images
    print(f"\n[5] Evaluating {n_images} images with default parameters...")
    print(f"  Parameters:")
    for key, val in DEFAULT_PARAMS.items():
        print(f"    {key:<20}: {val}")

    image_dir = Path(args.image_dir)
    results = []
    failed_count = 0

    for idx, img_name in enumerate(tqdm(selected_images, desc="Evaluating")):
        # Find image file
        img_path = image_dir / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = image_dir / f"{img_name}.png"

        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_name}")
            failed_count += 1
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Failed to load image: {img_name}")
            failed_count += 1
            continue

        # Get GT lines
        gt_lines = gt_data[img_name]

        try:
            # Evaluate
            score = evaluate_single_image(image, gt_lines, DEFAULT_PARAMS, yolo_detector)

            # Save result
            result = {
                "iteration": idx + 1,
                "image_name": img_name,
                "image_idx": image_names.index(img_name),
                "parameters": DEFAULT_PARAMS,
                "score": float(score),
                "acq_function": "Random",
                "acq_value": 0.0
            }
            results.append(result)

            # Save individual log
            log_file = output_dir / f"iter_{idx+1:03d}.json"
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"  [ERROR] Failed for {img_name}: {e}")
            failed_count += 1
            continue

    # 6. Summary
    print(f"\n[6] Evaluation Complete!")
    print(f"  Total evaluated: {len(results)}/{n_images}")
    print(f"  Failed: {failed_count}")

    # Save summary
    summary = {
        "total_images": n_images,
        "evaluated": len(results),
        "failed": failed_count,
        "seed": args.seed,
        "parameters": DEFAULT_PARAMS,
        "scores": {
            "mean": float(np.mean([r['score'] for r in results])),
            "std": float(np.std([r['score'] for r in results])),
            "min": float(np.min([r['score'] for r in results])),
            "max": float(np.max([r['score'] for r in results]))
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[7] Score Statistics:")
    print(f"  Mean:  {summary['scores']['mean']:.4f}")
    print(f"  Std:   {summary['scores']['std']:.4f}")
    print(f"  Min:   {summary['scores']['min']:.4f}")
    print(f"  Max:   {summary['scores']['max']:.4f}")

    print("\n" + "="*70)
    print("Next step:")
    print(f"  python analyze_clip_correlation.py \\")
    print(f"      --log_dir {args.output_dir} \\")
    print(f"      --clip_features environment_roi_worst_case_v2.json")
    print("\n  This will use {0} evaluated images for correlation analysis!".format(len(results)))
    print("="*70)


if __name__ == "__main__":
    main()
