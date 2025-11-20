"""
Validation script for best parameters from BoRisk optimization.
Evaluates optimized parameters on held-out validation set (246 images).
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from full_pipeline import detect_lines_in_roi
from yolo_detector import YOLODetector
from evaluation import evaluate_lp
from environment_independent import extract_environment_features


def load_validation_images(val_json_path, image_dir, gt_file):
    """Load validation image list and corresponding GT data."""
    # Load validation image names
    with open(val_json_path, 'r') as f:
        val_names = json.load(f)

    # Load ground truth
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    # Build validation dataset
    val_data = []
    image_dir = Path(image_dir)

    for img_name in val_names:
        # Find matching image file
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            candidate = image_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"Warning: Image not found: {img_name}")
            continue

        # Find GT
        gt_entry = None
        for gt in gt_data:
            if gt['image_name'] == img_name:
                gt_entry = gt
                break

        if gt_entry is None:
            print(f"Warning: GT not found: {img_name}")
            continue

        val_data.append({
            'image_path': str(img_path),
            'image_name': img_name,
            'gt_coords': gt_entry['coordinates']
        })

    return val_data


def evaluate_on_validation(params_dict, val_data, yolo_model_path, alpha=0.3):
    """
    Evaluate given parameters on validation set.

    Args:
        params_dict: Dictionary of parameters
        val_data: List of validation image data
        yolo_model_path: Path to YOLO model
        alpha: CVaR threshold

    Returns:
        dict: Results including CVaR, mean score, etc.
    """
    # Initialize YOLO detector
    yolo_detector = YOLODetector(model_path=yolo_model_path)

    # Extract parameters
    edgeThresh1 = params_dict['edgeThresh1']
    simThresh1 = params_dict['simThresh1']
    pixelRatio1 = params_dict['pixelRatio1']
    edgeThresh2 = params_dict['edgeThresh2']
    simThresh2 = params_dict['simThresh2']
    pixelRatio2 = params_dict['pixelRatio2']
    ransac_weight_q = params_dict['ransac_weight_q']
    ransac_weight_qg = params_dict['ransac_weight_qg']

    # Evaluate all images
    scores = []
    env_features = []
    failed_images = []

    print(f"\nEvaluating {len(val_data)} validation images...")

    for data in tqdm(val_data, desc="Validation"):
        img_path = data['image_path']
        img_name = data['image_name']
        gt_coords = data['gt_coords']

        try:
            # YOLO detection
            results = yolo_detector.detect(img_path, save=False, verbose=False)

            if results is None or len(results) == 0:
                scores.append(0.0)
                failed_images.append(img_name)
                continue

            result = results[0]
            orig_img = result.orig_img

            if result.boxes is None or len(result.boxes) == 0:
                scores.append(0.0)
                failed_images.append(img_name)
                continue

            # AirLine detection
            detected_coords = detect_lines_in_roi(
                orig_img,
                result,
                edgeThresh1, simThresh1, pixelRatio1,
                edgeThresh2, simThresh2, pixelRatio2,
                ransac_weight_q, ransac_weight_qg,
                debug=False
            )

            # Evaluation
            score = evaluate_lp(
                detected_coords,
                orig_img,
                image_name=img_name,
                threshold=20.0,
                debug=False
            )
            scores.append(score)

            # Extract environment features
            env = extract_environment_features(orig_img)
            env_features.append(env)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            scores.append(0.0)
            failed_images.append(img_name)

    # Calculate metrics
    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Calculate CVaR
    sorted_scores = np.sort(scores)
    n_worst = max(1, int(alpha * len(scores)))
    cvar = np.mean(sorted_scores[:n_worst])

    # Environment statistics
    env_features = np.array(env_features) if env_features else None

    results = {
        'n_images': len(val_data),
        'n_evaluated': len(scores),
        'n_failed': len(failed_images),
        'mean_score': float(mean_score),
        'std_score': float(std_score),
        'cvar': float(cvar),
        'alpha': alpha,
        'scores': scores.tolist(),
        'failed_images': failed_images
    }

    if env_features is not None:
        results['environment_stats'] = {
            'brightness_mean': float(np.mean(env_features[:, 0])),
            'contrast_mean': float(np.mean(env_features[:, 1])),
            'edge_density_mean': float(np.mean(env_features[:, 2])),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate best parameters')
    parser.add_argument('--params_file', type=str, required=True,
                        help='Path to iteration JSON file (e.g., iter_086.json)')
    parser.add_argument('--val_json', type=str, default='validation_images.json',
                        help='Path to validation images JSON')
    parser.add_argument('--image_dir', type=str, default='../dataset/images/for_BO',
                        help='Image directory')
    parser.add_argument('--gt_file', type=str, default='../dataset/ground_truth_merged.json',
                        help='Ground truth JSON file')
    parser.add_argument('--yolo_model', type=str, default='models/best.pt',
                        help='YOLO model path')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='CVaR alpha threshold')
    parser.add_argument('--output', type=str, default='validation_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    # Load parameters
    print(f"Loading parameters from {args.params_file}...")
    with open(args.params_file, 'r') as f:
        iter_data = json.load(f)

    params = iter_data['parameters']
    print(f"Parameters from iteration {iter_data['iteration']}:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    # Load validation data
    print(f"\nLoading validation data from {args.val_json}...")
    val_data = load_validation_images(args.val_json, args.image_dir, args.gt_file)
    print(f"Loaded {len(val_data)} validation images")

    # Evaluate
    results = evaluate_on_validation(
        params,
        val_data,
        args.yolo_model,
        alpha=args.alpha
    )

    # Add metadata
    results['params_source'] = args.params_file
    results['iteration'] = iter_data['iteration']
    results['training_cvar'] = iter_data['cvar']
    results['parameters'] = params

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Parameters: Iteration {iter_data['iteration']}")
    print(f"Training CVaR: {iter_data['cvar']:.4f}")
    print(f"\nValidation Set ({results['n_images']} images):")
    print(f"  Mean Score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"  CVaR (α={args.alpha}): {results['cvar']:.4f}")
    print(f"  Failed: {results['n_failed']} images")
    print(f"\nPerformance Gap:")
    print(f"  CVaR Gap: {results['cvar'] - iter_data['cvar']:.4f}")
    print(f"  (Negative gap indicates generalization)")
    print("="*60)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
