#!/usr/bin/env python
"""최소 테스트"""
import sys
print("Step 1: Parse args")
sys.argv = ['optimization.py', '--iterations', '1', '--n_initial', '1', '--alpha', '0.3']

print("Step 2: Import optimization")
import optimization

print("Step 3: Load data")
from pathlib import Path
import json

dataset_path = Path("dataset/images/test")
gt_path = Path("dataset/ground_truth.json")

with open(gt_path, 'r') as f:
    ground_truth = json.load(f)

images_data = []
for img_name, gt_data in list(ground_truth.items())[:2]:  # 2개만
    img_path = dataset_path / f"{img_name}.jpg"
    if img_path.exists():
        images_data.append({
            'name': img_name,
            'path': str(img_path),
            'gt': gt_data
        })

print(f"Step 4: Loaded {len(images_data)} images")

print("Step 5: Initialize YOLO")
from yolo_detector import YOLODetector
yolo = YOLODetector("models/best.pt")

print("Step 6: Test objective function")
import torch
X = torch.tensor([[-3.0, 0.98, 0.05, 1.0, 0.75, 0.05, 0.5, 0.5, 5]],
                 dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
try:
    score = optimization.objective_function(X, images_data, yolo, alpha=0.3, verbose=True)
    print(f"✓ CVaR = {score}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
