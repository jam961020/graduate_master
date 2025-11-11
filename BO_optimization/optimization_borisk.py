"""
Risk-aware Bayesian Optimization using BoTorch CVaR
Based on: Bayesian Optimization of Risk Measures (Cakmak et al., NeurIPS 2020)

Core idea: Optimize ρ[F(x, W)] where
- x: design parameters (AirLine algorithm parameters)
- W: environmental variable (image augmentation)
- F(x, W): detection quality score
- ρ: CVaR risk measure
"""
import torch
import numpy as np
import cv2
import json
import datetime
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.risk_measures import CVaR
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import AppendFeatures
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector
from evaluation import evaluate_quality

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# Design parameters bounds (x)
BOUNDS_X = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01],  # lower
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15]      # upper
], dtype=DTYPE, device=DEVICE)

# Environmental variable bounds (W)
# W represents augmentation parameters: [noise_sigma, brightness, contrast]
BOUNDS_W = torch.tensor([
    [2.0, 0.85, 0.9],   # lower
    [8.0, 1.15, 1.1]    # upper
], dtype=DTYPE, device=DEVICE)

# Combined bounds [x, w]
BOUNDS = torch.cat([BOUNDS_X, BOUNDS_W], dim=-1)


def augment_image_with_params(image, w):
    """
    Apply augmentation with specific parameters w = [noise_sigma, brightness, contrast]
    
    Args:
        image: Original image (np.ndarray)
        w: Augmentation parameters [noise_sigma, brightness, contrast]
    
    Returns:
        Augmented image (np.ndarray)
    """
    noise_sigma, brightness, contrast = w
    
    aug = image.copy().astype(np.float32)
    
    # 1. Gaussian noise
    noise = np.random.normal(0, noise_sigma, image.shape)
    aug = aug + noise
    
    # 2. Brightness
    aug = aug * brightness
    
    # 3. Contrast
    aug = (aug - 128) * contrast + 128
    
    # Clip and convert
    aug = np.clip(aug, 0, 255).astype(np.uint8)
    
    return aug


def load_dataset(image_dir, gt_file, complete_only=False):
    """
    Load dataset (original images only, no pre-augmentation)
    
    Args:
        image_dir: Directory containing images
        gt_file: Ground truth JSON file
        complete_only: If True, only load images with all 12 coordinates
    
    Returns:
        List of image data dicts
    """
    image_dir = Path(image_dir)
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    images_data = []
    
    for img_name, data in gt_data.items():
        possible_paths = [
            image_dir / f"{img_name}.jpg",
            image_dir / f"{img_name}.png",
        ]
        
        img_path = None
        for p in possible_paths:
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        gt_coords = data.get('coordinates', data)
        
        # Filter complete data only
        if complete_only:
            required_keys = [
                'longi_left_lower_x', 'longi_left_lower_y',
                'longi_right_lower_x', 'longi_right_lower_y',
                'longi_left_upper_x', 'longi_left_upper_y',
                'longi_right_upper_x', 'longi_right_upper_y',
                'collar_left_lower_x', 'collar_left_lower_y',
                'collar_left_upper_x', 'collar_left_upper_y'
            ]
            if not all(gt_coords.get(k, 0) != 0 for k in required_keys):
                continue
        
        img_data = {
            'name': img_name,
            'image': image,
            'gt_coords': gt_coords,
            'environment': data.get('environment', {})
        }
        images_data.append(img_data)
    
    return images_data


def evaluate_F_xw(x, w, images_data, yolo_detector, metric="lp"):
    """
    Evaluate F(x, w) for a single (x, w) pair
    
    Args:
        x: Design parameters [6,] tensor
        w: Environmental parameters [3,] tensor
        images_data: List of original images
        yolo_detector: YOLODetector instance
        metric: "lp" or "endpoint"
    
    Returns:
        Average score across all images under this (x, w)
    """
    params = {
        'edgeThresh1': x[0].item(),
        'simThresh1': x[1].item(),
        'pixelRatio1': x[2].item(),
        'edgeThresh2': x[3].item(),
        'simThresh2': x[4].item(),
        'pixelRatio2': x[5].item(),
    }
    
    w_np = w.cpu().numpy()
    scores = []
    
    for img_data in images_data:
        try:
            # Apply augmentation with specific w
            image_aug = augment_image_with_params(img_data['image'], w_np)
            
            # Detect with parameters x
            detected_coords = detect_with_full_pipeline(image_aug, params, yolo_detector)
            
            # Evaluate
            score = evaluate_quality(detected_coords, img_data['image'], 
                                    img_data['name'], metric=metric)
            scores.append(score)
            
        except Exception as e:
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0


def objective_with_environmental_var(X, images_data, yolo_detector, 
                                    n_w_samples=16, metric="lp", alpha=0.7):
    """
    Objective function: CVaR[F(x, W)] 
    
    For each x in X, sample W multiple times and compute CVaR
    
    Args:
        X: [batch_size, 6] design parameters
        images_data: Original images
        yolo_detector: YOLODetector instance
        n_w_samples: Number of W samples for CVaR estimation
        metric: "lp" or "endpoint"
        alpha: CVaR risk level (probability mass of worst outcomes)
    
    Returns:
        Y: [batch_size, 1] CVaR values
    """
    batch_size = X.shape[0]
    Y = []
    
    for i in range(batch_size):
        x = X[i]
        
        # Sample W from uniform distribution
        w_samples = torch.rand(n_w_samples, 3, dtype=DTYPE, device=DEVICE)
        w_samples = BOUNDS_W[0] + (BOUNDS_W[1] - BOUNDS_W[0]) * w_samples
        
        # Evaluate F(x, w) for each w sample
        f_values = []
        for w in w_samples:
            f_xw = evaluate_F_xw(x, w, images_data, yolo_detector, metric)
            f_values.append(f_xw)
        
        f_values = np.array(f_values)
        
        # Compute CVaR (α-tail mean)
        n_worst = max(1, int(len(f_values) * alpha))
        worst_values = np.sort(f_values)[:n_worst]
        cvar = np.mean(worst_values)
        
        Y.append(cvar)
        
        print(f"  x[{i}]: CVaR={cvar:.4f} (mean={f_values.mean():.4f}, "
              f"min={f_values.min():.4f})")
    
    return torch.tensor(Y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)


def optimize_with_borisk(images_data, yolo_detector, metric="lp",
                        n_iterations=30, n_initial=10, n_w_samples=16, alpha=0.7):
    """
    Main optimization loop using BoRisk framework
    
    Args:
        images_data: Original images (no augmentation)
        yolo_detector: YOLODetector instance
        metric: "lp" or "endpoint"
        n_iterations: Number of BO iterations
        n_initial: Number of initial samples
        n_w_samples: Number of W samples for CVaR estimation
        alpha: CVaR risk level
    
    Returns:
        best_x: Best design parameters
        history: CVaR history
        all_Y: All observed CVaR values
    """
    print("\n" + "="*60)
    print(f"BoRisk: CVaR-based Bayesian Optimization")
    print(f"Metric: {metric.upper()}, α={alpha}, W samples={n_w_samples}")
    print(f"Images: {len(images_data)}")
    print("="*60)
    
    # Initial samples (x only, W will be sampled during evaluation)
    print(f"\nGenerating {n_initial} initial samples...")
    
    sobol = SobolEngine(dimension=6, scramble=True)
    X_init = sobol.draw(n_initial).to(dtype=DTYPE, device=DEVICE)
    X_init = BOUNDS_X[0] + (BOUNDS_X[1] - BOUNDS_X[0]) * X_init
    
    print("Evaluating initial samples...")
    Y_init = objective_with_environmental_var(
        X_init, images_data, yolo_detector, 
        n_w_samples=n_w_samples, metric=metric, alpha=alpha
    )
    
    X, Y = X_init, Y_init
    best_observed = [Y.max().item()]
    
    print(f"\nInitial best CVaR: {best_observed[0]:.4f}\n")
    print("Starting BO iterations...")
    
    # BO loop
    for iteration in range(n_iterations):
        # Fit GP model
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # Acquisition function (standard EI for now)
        # Note: For full BoRisk, we'd use qMultiFidelityKnowledgeGradient
        # with CVaR objective, but that's more complex
        EI = qExpectedImprovement(gp, Y.max())
        
        # Optimize acquisition function
        candidate, acq_value = optimize_acqf(
            EI,
            bounds=BOUNDS_X,
            q=1,
            num_restarts=10,
            raw_samples=512
        )
        
        # Evaluate candidate
        print(f"\nIter {iteration+1}/{n_iterations}:")
        new_y = objective_with_environmental_var(
            candidate, images_data, yolo_detector,
            n_w_samples=n_w_samples, metric=metric, alpha=alpha
        )
        
        # Update data
        X = torch.cat([X, candidate])
        Y = torch.cat([Y, new_y])
        
        best_observed.append(Y.max().item())
        
        improvement = best_observed[-1] - best_observed[-2]
        print(f"  → New CVaR={new_y.item():.4f}, "
              f"Best={best_observed[-1]:.4f} ({improvement:+.4f})")
    
    # Best parameters
    best_idx = Y.argmax()
    best_x = X[best_idx]
    
    print("\n" + "="*60)
    print("BO Optimization Complete")
    print("="*60)
    
    return best_x, best_observed, Y


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="BoRisk-based optimization")
    parser.add_argument("--data_dir", required=True, help="Image directory")
    parser.add_argument("--gt_file", required=True, help="Ground truth JSON")
    parser.add_argument("--yolo_model", required=True, help="YOLO model path")
    parser.add_argument("--metric", default="lp", choices=["lp", "endpoint"])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--n_initial", type=int, default=10)
    parser.add_argument("--n_w_samples", type=int, default=16, 
                       help="Number of W samples for CVaR")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="CVaR risk level (0-1)")
    parser.add_argument("--complete_only", action="store_true")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading dataset from {args.data_dir}...")
    images_data = load_dataset(args.data_dir, args.gt_file, args.complete_only)
    
    if len(images_data) == 0:
        print("ERROR: No valid images found!")
        sys.exit(1)
    
    print(f"Loaded {len(images_data)} images")
    
    # Initialize YOLO
    print(f"\nInitializing YOLO detector...")
    yolo_detector = YOLODetector(args.yolo_model)
    
    # Run optimization
    best_params, history, all_Y = optimize_with_borisk(
        images_data,
        yolo_detector,
        metric=args.metric,
        n_iterations=args.iterations,
        n_initial=args.n_initial,
        n_w_samples=args.n_w_samples,
        alpha=args.alpha
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nBest parameters ({args.metric.upper()}, α={args.alpha}):")
    print(f"  edgeThresh1:  {best_params[0]:7.2f}")
    print(f"  simThresh1:   {best_params[1]:7.4f}")
    print(f"  pixelRatio1:  {best_params[2]:7.4f}")
    print(f"  edgeThresh2:  {best_params[3]:7.2f}")
    print(f"  simThresh2:   {best_params[4]:7.4f}")
    print(f"  pixelRatio2:  {best_params[5]:7.4f}")
    
    print(f"\nPerformance:")
    print(f"  Final CVaR:    {history[-1]:.4f}")
    print(f"  Initial CVaR:  {history[0]:.4f}")
    print(f"  Improvement:   {(history[-1] - history[0]) / (history[0] + 1e-6) * 100:+.1f}%")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"borisk_{args.metric}_alpha{args.alpha}_{timestamp}.json"
    
    result_data = {
        "method": "BoRisk_CVaR",
        "metric": args.metric,
        "alpha": args.alpha,
        "n_w_samples": args.n_w_samples,
        "n_images": len(images_data),
        "iterations": args.iterations,
        "best_params": {
            "edgeThresh1": float(best_params[0]),
            "simThresh1": float(best_params[1]),
            "pixelRatio1": float(best_params[2]),
            "edgeThresh2": float(best_params[3]),
            "simThresh2": float(best_params[4]),
            "pixelRatio2": float(best_params[5]),
        },
        "history": [float(x) for x in history],
        "final_cvar": float(history[-1]),
        "improvement": float((history[-1] - history[0]) / (history[0] + 1e-6) * 100)
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\nResults saved: {result_file}")
    print("="*60)