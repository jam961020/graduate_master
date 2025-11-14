"""
Metric 비교 분석 스크립트

기존 결과 파일들을 로드해서 다양한 metric으로 재평가하고 비교
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def load_iteration_data(log_dir: str) -> List[Dict]:
    """
    iteration 결과 로드 (detected_coords, gt_coords 포함)
    """
    log_path = Path(log_dir)
    iter_files = sorted(log_path.glob("iter_*.json"))

    data = []
    for f in iter_files:
        with open(f) as fp:
            d = json.load(fp)
            data.append(d)

    return data


# ============================================================
# Metric 1: 현재 방식 (기울기 차이 + 거리)
# ============================================================
def metric_current(detected_coords, gt_coords, image_size=(640, 480),
                   direction_weight=0.6, distance_weight=0.4):
    """
    현재 구현 (optimization.py:52-147)

    문제점:
    1. distance_threshold = 40px (너무 작음!)
    2. direction_sim = 1/(1+slope_diff) (비선형, 가파름)
    3. 1개 실패 시 평균이 크게 하락
    """
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('fillet_left', 'longi_left_lower_x', 'longi_left_lower_y',
                       'collar_left_lower_x', 'collar_left_lower_y'),
        ('fillet_right', 'collar_left_lower_x', 'collar_left_lower_y',
                        'longi_right_lower_x', 'longi_right_lower_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]

    diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
    distance_threshold = diagonal * 0.05  # 40px

    line_scores = []

    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)

        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue

        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue

        def to_line_eq(x1, y1, x2, y2):
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            norm = np.sqrt(A**2 + B**2) + 1e-6
            return A/norm, B/norm, C/norm

        A_gt, B_gt, C_gt = to_line_eq(gt_x1, gt_y1, gt_x2, gt_y2)
        A_det, B_det, C_det = to_line_eq(det_x1, det_y1, det_x2, det_y2)

        # 기울기 차이
        slope_gt = -A_gt / B_gt if abs(B_gt) > 1e-6 else 1e6
        slope_det = -A_det / B_det if abs(B_det) > 1e-6 else 1e6
        slope_diff = abs(slope_gt - slope_det)
        direction_sim = 1.0 / (1.0 + slope_diff)

        # 거리
        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = max(0.0, 1.0 - (parallel_dist / distance_threshold))

        line_score = direction_weight * direction_sim + distance_weight * distance_sim
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


# ============================================================
# Metric 2: Exponential Decay (거리 패널티 완화)
# ============================================================
def metric_exponential(detected_coords, gt_coords, image_size=(640, 480),
                       direction_weight=0.6, distance_weight=0.4,
                       distance_scale=100.0):
    """
    개선 1: 거리 패널티를 exponential decay로 변경

    distance_sim = exp(-distance / scale)
    - scale=100px: 100px 오차 → 0.37점
    - scale=50px:  100px 오차 → 0.14점

    부드럽게 감소 → 부분 점수 부여
    """
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('fillet_left', 'longi_left_lower_x', 'longi_left_lower_y',
                       'collar_left_lower_x', 'collar_left_lower_y'),
        ('fillet_right', 'collar_left_lower_x', 'collar_left_lower_y',
                        'longi_right_lower_x', 'longi_right_lower_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]

    line_scores = []

    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)

        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue

        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue

        def to_line_eq(x1, y1, x2, y2):
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            norm = np.sqrt(A**2 + B**2) + 1e-6
            return A/norm, B/norm, C/norm

        A_gt, B_gt, C_gt = to_line_eq(gt_x1, gt_y1, gt_x2, gt_y2)
        A_det, B_det, C_det = to_line_eq(det_x1, det_y1, det_x2, det_y2)

        # 기울기 차이 (동일)
        slope_gt = -A_gt / B_gt if abs(B_gt) > 1e-6 else 1e6
        slope_det = -A_det / B_det if abs(B_det) > 1e-6 else 1e6
        slope_diff = abs(slope_gt - slope_det)
        direction_sim = 1.0 / (1.0 + slope_diff)

        # 거리: Exponential decay
        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = np.exp(-parallel_dist / distance_scale)

        line_score = direction_weight * direction_sim + distance_weight * distance_sim
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


# ============================================================
# Metric 3: Angle-based (각도 차이)
# ============================================================
def metric_angle_based(detected_coords, gt_coords, image_size=(640, 480),
                       direction_weight=0.6, distance_weight=0.4,
                       distance_scale=100.0):
    """
    개선 2: 기울기 차이 → 각도 차이로 변경

    angle_sim = cos(angle_diff) 또는 exp(-angle_diff / scale)
    더 직관적이고 해석 가능
    """
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('fillet_left', 'longi_left_lower_x', 'longi_left_lower_y',
                       'collar_left_lower_x', 'collar_left_lower_y'),
        ('fillet_right', 'collar_left_lower_x', 'collar_left_lower_y',
                        'longi_right_lower_x', 'longi_right_lower_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]

    line_scores = []

    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)

        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue

        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue

        def to_line_eq(x1, y1, x2, y2):
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            norm = np.sqrt(A**2 + B**2) + 1e-6
            return A/norm, B/norm, C/norm

        A_gt, B_gt, C_gt = to_line_eq(gt_x1, gt_y1, gt_x2, gt_y2)
        A_det, B_det, C_det = to_line_eq(det_x1, det_y1, det_x2, det_y2)

        # 각도 차이 (코사인 유사도)
        # A*A' + B*B' = cos(theta) (이미 정규화됨)
        cos_theta = abs(A_gt * A_det + B_gt * B_det)  # abs: 방향 무시
        direction_sim = cos_theta  # [0, 1]

        # 거리: Exponential decay
        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = np.exp(-parallel_dist / distance_scale)

        line_score = direction_weight * direction_sim + distance_weight * distance_sim
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


# ============================================================
# Metric 4: Endpoint Distance (끝점 거리)
# ============================================================
def metric_endpoint_distance(detected_coords, gt_coords, image_size=(640, 480),
                             distance_scale=50.0):
    """
    개선 3: 끝점 거리 기반 평가

    단순히 2개 끝점의 Euclidean 거리 평균
    - 직관적
    - 해석 가능
    - 선 방정식 불필요
    """
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('fillet_left', 'longi_left_lower_x', 'longi_left_lower_y',
                       'collar_left_lower_x', 'collar_left_lower_y'),
        ('fillet_right', 'collar_left_lower_x', 'collar_left_lower_y',
                        'longi_right_lower_x', 'longi_right_lower_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]

    line_scores = []

    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)

        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue

        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue

        # 끝점 거리
        dist1 = np.sqrt((gt_x1 - det_x1)**2 + (gt_y1 - det_y1)**2)
        dist2 = np.sqrt((gt_x2 - det_x2)**2 + (gt_y2 - det_y2)**2)
        avg_dist = (dist1 + dist2) / 2

        # Exponential decay
        line_score = np.exp(-avg_dist / distance_scale)
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


# ============================================================
# Metric 5: Weighted (선별 가중치)
# ============================================================
def metric_weighted(detected_coords, gt_coords, image_size=(640, 480),
                    distance_scale=100.0):
    """
    개선 4: 선별 중요도 가중치 적용

    longi > collar > fillet
    세로선이 더 중요 (구조적 핵심)
    """
    line_definitions = [
        # (name, x1, y1, x2, y2, weight)
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y', 1.5),  # 중요!
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y', 1.5),  # 중요!
        ('fillet_left', 'longi_left_lower_x', 'longi_left_lower_y',
                       'collar_left_lower_x', 'collar_left_lower_y', 0.8),
        ('fillet_right', 'collar_left_lower_x', 'collar_left_lower_y',
                        'longi_right_lower_x', 'longi_right_lower_y', 0.8),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y', 1.2)
    ]

    line_scores = []
    weights = []

    for name, x1_key, y1_key, x2_key, y2_key, weight in line_definitions:
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)

        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue

        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            weights.append(weight)
            continue

        def to_line_eq(x1, y1, x2, y2):
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            norm = np.sqrt(A**2 + B**2) + 1e-6
            return A/norm, B/norm, C/norm

        A_gt, B_gt, C_gt = to_line_eq(gt_x1, gt_y1, gt_x2, gt_y2)
        A_det, B_det, C_det = to_line_eq(det_x1, det_y1, det_x2, det_y2)

        # 각도 (코사인 유사도)
        cos_theta = abs(A_gt * A_det + B_gt * B_det)

        # 거리
        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = np.exp(-parallel_dist / distance_scale)

        line_score = 0.6 * cos_theta + 0.4 * distance_sim
        line_scores.append(line_score)
        weights.append(weight)

    if len(line_scores) == 0:
        return 0.0

    # 가중 평균
    return float(np.average(line_scores, weights=weights))


# ============================================================
# 분석 함수
# ============================================================
def analyze_metrics(log_dir: str, output_dir: str):
    """
    모든 iteration에 대해 다양한 metric 계산 및 비교
    """
    print("="*70)
    print("Metric 비교 분석")
    print("="*70)

    data = load_iteration_data(log_dir)
    print(f"\n로드된 iterations: {len(data)}")

    # Ground truth 로드
    gt_file = Path("../dataset/ground_truth_auto.json")
    with open(gt_file) as f:
        gt_data = json.load(f)

    results = {
        'iteration': [],
        'score_original': [],
        'metric_current': [],
        'metric_exponential': [],
        'metric_angle': [],
        'metric_endpoint': [],
        'metric_weighted': []
    }

    print("\n계산 중...")
    for iter_data in data:
        iteration = iter_data['iteration']
        score_original = iter_data['score']

        # detected_coords 가져오기 (없으면 skip)
        if 'detected_coords' not in iter_data:
            continue

        detected = iter_data['detected_coords']

        # GT 가져오기
        img_idx = iter_data['image_idx']
        img_name = f"test_{img_idx:03d}.jpg"
        if img_name not in gt_data:
            continue

        gt = gt_data[img_name]

        # 각 metric 계산
        m1 = metric_current(detected, gt)
        m2 = metric_exponential(detected, gt, distance_scale=100.0)
        m3 = metric_angle_based(detected, gt, distance_scale=100.0)
        m4 = metric_endpoint_distance(detected, gt, distance_scale=50.0)
        m5 = metric_weighted(detected, gt, distance_scale=100.0)

        results['iteration'].append(iteration)
        results['score_original'].append(score_original)
        results['metric_current'].append(m1)
        results['metric_exponential'].append(m2)
        results['metric_angle'].append(m3)
        results['metric_endpoint'].append(m4)
        results['metric_weighted'].append(m5)

        if iteration <= 10 or iteration % 10 == 0:
            print(f"Iter {iteration:3d}: Original={score_original:.3f}, "
                  f"Current={m1:.3f}, Exp={m2:.3f}, Angle={m3:.3f}, "
                  f"Endpoint={m4:.3f}, Weighted={m5:.3f}")

    # 통계 분석
    print("\n" + "="*70)
    print("통계 요약")
    print("="*70)

    metrics_dict = {
        'Original Score': results['score_original'],
        'Current': results['metric_current'],
        'Exponential': results['metric_exponential'],
        'Angle-based': results['metric_angle'],
        'Endpoint': results['metric_endpoint'],
        'Weighted': results['metric_weighted']
    }

    for name, values in metrics_dict.items():
        vals = np.array(values)
        print(f"\n{name}:")
        print(f"  Mean: {vals.mean():.4f}")
        print(f"  Std:  {vals.std():.4f}")
        print(f"  Min:  {vals.min():.4f}")
        print(f"  Max:  {vals.max():.4f}")
        print(f"  Range: {vals.max() - vals.min():.4f}")

    # 시각화
    print("\n시각화 생성 중...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metric Comparison Analysis', fontsize=16)

    iterations = results['iteration']

    # 1. 전체 추이
    ax = axes[0, 0]
    for name, values in metrics_dict.items():
        ax.plot(iterations, values, label=name, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Score Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 분포 (Histogram)
    ax = axes[0, 1]
    for name, values in metrics_dict.items():
        ax.hist(values, bins=20, alpha=0.5, label=name)
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Box plot
    ax = axes[0, 2]
    ax.boxplot([v for v in metrics_dict.values()],
               labels=[k for k in metrics_dict.keys()])
    ax.set_ylabel('Score')
    ax.set_title('Score Distributions (Box Plot)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # 4-6. Scatter plots (Original vs New Metrics)
    scatter_pairs = [
        ('Exponential', results['metric_exponential']),
        ('Angle-based', results['metric_angle']),
        ('Weighted', results['metric_weighted'])
    ]

    for idx, (name, values) in enumerate(scatter_pairs):
        ax = axes[1, idx]
        ax.scatter(results['score_original'], values, alpha=0.5)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Original Score')
        ax.set_ylabel(f'{name} Metric')
        ax.set_title(f'Original vs {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Correlation
        corr = np.corrcoef(results['score_original'], values)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / "metric_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: {fig_path}")

    # JSON 저장
    json_path = output_path / "metric_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"결과 저장: {json_path}")

    plt.show()

    print("\n" + "="*70)
    print("분석 완료!")
    print("="*70)

    return results


if __name__ == "__main__":
    log_dir = "logs/run_20251114_172045"
    output_dir = "results/metric_analysis"

    results = analyze_metrics(log_dir, output_dir)
