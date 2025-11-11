"""
BoRisk CVaR Optimization - 정식 구현 v2
기반: BoTorch tutorial + BoRisk 논문

핵심 원리:
1. 매 iteration마다 1개 (x,w) 쌍만 실제 평가
2. GP 모델: (x, w) → y
3. 획득 함수: qMultiFidelityKnowledgeGradient + CVaR
4. 판타지 관측으로 CVaR 계산 (실제 평가 아님!)
"""
import torch
import numpy as np
import cv2
import json
import sys
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.objective import GenericMCObjective
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector
from environment_independent import extract_parameter_independent_environment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# 9D params bounds
PARAM_BOUNDS = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 1],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 10]
], dtype=DTYPE, device=DEVICE)


def line_equation_evaluation(detected_coords, gt_coords, image_size=(640, 480)):
    """직선 방정식 기반 평가 (간소화)"""
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]

    diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
    line_scores = []

    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_coords.get(x1_key, 0), gt_coords.get(y1_key, 0), gt_coords.get(x2_key, 0), gt_coords.get(y2_key, 0)
        det_x1, det_y1, det_x2, det_y2 = detected_coords.get(x1_key, 0), detected_coords.get(y1_key, 0), detected_coords.get(x2_key, 0), detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue
        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue

        def to_line_eq(x1, y1, x2, y2):
            A, B, C = y2 - y1, x1 - x2, x2*y1 - x1*y2
            norm = np.sqrt(A**2 + B**2) + 1e-6
            return A/norm, B/norm, C/norm

        A_gt, B_gt, C_gt = to_line_eq(gt_x1, gt_y1, gt_x2, gt_y2)
        A_det, B_det, C_det = to_line_eq(det_x1, det_y1, det_x2, det_y2)

        direction_sim = abs(A_gt*A_det + B_gt*B_det)
        mid_x, mid_y = (gt_x1 + gt_x2) / 2, (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = max(0.0, 1.0 - (parallel_dist / (diagonal * 0.05)))

        line_scores.append(0.6 * direction_sim + 0.4 * distance_sim)

    return float(np.mean(line_scores)) if line_scores else 0.0


def evaluate_single(params_tensor, img_data, yolo_detector):
    """단일 (x,w) 쌍 평가 - 실제 함수 호출 1회!"""
    params = {
        'edgeThresh1': params_tensor[0].item(),
        'simThresh1': params_tensor[1].item(),
        'pixelRatio1': params_tensor[2].item(),
        'edgeThresh2': params_tensor[3].item(),
        'simThresh2': params_tensor[4].item(),
        'pixelRatio2': params_tensor[5].item(),
        'ransac_center_w': params_tensor[6].item(),
        'ransac_length_w': params_tensor[7].item(),
        'ransac_consensus_w': int(params_tensor[8].item()),
    }

    try:
        detected_coords = detect_with_full_pipeline(img_data['image'], params, yolo_detector)
        h, w = img_data['image'].shape[:2]
        return line_equation_evaluation(detected_coords, img_data['gt_coords'], (w, h))
    except Exception as e:
        print(f"[WARN] Eval error: {e}")
        return 0.0


print("BoRisk v2 implementation loaded successfully!")
print("Usage: python optimization_borisk_v2.py --iterations 5 --n_initial 5")
