"""
Metric 비교 분석 (Lite 버전)

선택된 iteration의 파라미터로 재실행해서 metric 비교
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from typing import List, Dict

# optimization.py의 함수들 임포트
sys.path.insert(0, str(Path(__file__).parent))
from optimization import line_equation_evaluation, load_dataset
import torch


# ============================================================
# 새로운 Metric들
# ============================================================
def metric_exponential(detected_coords, gt_coords, image_size=(640, 480),
                       direction_weight=0.6, distance_weight=0.4,
                       distance_scale=100.0):
    """거리 패널티를 exponential decay로 변경"""
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

        slope_gt = -A_gt / B_gt if abs(B_gt) > 1e-6 else 1e6
        slope_det = -A_det / B_det if abs(B_det) > 1e-6 else 1e6
        slope_diff = abs(slope_gt - slope_det)
        direction_sim = 1.0 / (1.0 + slope_diff)

        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = np.exp(-parallel_dist / distance_scale)  # ← Exponential!

        line_score = direction_weight * direction_sim + distance_weight * distance_sim
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


def metric_angle_based(detected_coords, gt_coords, image_size=(640, 480),
                       direction_weight=0.6, distance_weight=0.4,
                       distance_scale=100.0):
    """기울기 차이 → 각도 차이 (코사인 유사도)"""
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

        # 각도 (코사인 유사도)
        cos_theta = abs(A_gt * A_det + B_gt * B_det)
        direction_sim = cos_theta  # ← Angle-based!

        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = np.exp(-parallel_dist / distance_scale)

        line_score = direction_weight * direction_sim + distance_weight * distance_sim
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


def metric_endpoint_distance(detected_coords, gt_coords, image_size=(640, 480),
                             distance_scale=50.0):
    """끝점 거리 기반 평가 (가장 직관적)"""
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

        line_score = np.exp(-avg_dist / distance_scale)
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


def metric_weighted(detected_coords, gt_coords, image_size=(640, 480),
                    distance_scale=100.0):
    """선별 가중치 적용 (longi > collar > fillet)"""
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y', 1.5),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y', 1.5),
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

        cos_theta = abs(A_gt * A_det + B_gt * B_det)

        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)
        distance_sim = np.exp(-parallel_dist / distance_scale)

        line_score = 0.6 * cos_theta + 0.4 * distance_sim
        line_scores.append(line_score)
        weights.append(weight)

    if len(line_scores) == 0:
        return 0.0

    return float(np.average(line_scores, weights=weights))


# ============================================================
# 분석 함수
# ============================================================
def select_representative_iterations(log_dir: str, n_samples: int = 10):
    """
    대표 iteration 선택 (best, worst, median, random)
    """
    log_path = Path(log_dir)
    iter_files = sorted(log_path.glob("iter_*.json"))

    data = []
    for f in iter_files:
        with open(f) as fp:
            d = json.load(fp)
            data.append(d)

    # Score 기준 정렬
    data_sorted = sorted(data, key=lambda x: x['score'])

    # 선택
    selected = []

    # Best 3
    selected.extend(data_sorted[-3:])

    # Worst 3
    selected.extend(data_sorted[:3])

    # Median 2
    mid = len(data_sorted) // 2
    selected.extend([data_sorted[mid-1], data_sorted[mid]])

    # Random 2
    import random
    remaining = [d for d in data if d not in selected]
    if len(remaining) >= 2:
        selected.extend(random.sample(remaining, 2))

    print(f"\n선택된 {len(selected)} iterations:")
    for d in selected:
        print(f"  Iter {d['iteration']:3d}: Score={d['score']:.4f}, CVaR={d['cvar']:.4f}")

    return selected


def analyze_selected_iterations(selected_data: List[Dict],
                                images_data: List[Dict],
                                output_dir: str):
    """
    선택된 iteration을 다양한 metric으로 평가
    """
    print("\n" + "="*70)
    print("Metric 비교 분석 (선택된 iterations)")
    print("="*70)

    results = {
        'iteration': [],
        'score_original': [],
        'cvar_original': [],
        'metric_current': [],
        'metric_exponential_50': [],
        'metric_exponential_100': [],
        'metric_exponential_200': [],
        'metric_angle_100': [],
        'metric_endpoint_30': [],
        'metric_endpoint_50': [],
        'metric_endpoint_100': [],
        'metric_weighted': []
    }

    # GT 로드
    gt_file = Path("../dataset/ground_truth_auto.json")
    with open(gt_file) as f:
        gt_data = json.load(f)

    # 각 iteration 재실행
    print("\n재실행 및 metric 계산...")

    from full_pipeline import detect_lines_with_airline

    for iter_data in selected_data:
        iteration = iter_data['iteration']
        params = iter_data['parameters']
        score_orig = iter_data['score']
        cvar_orig = iter_data['cvar']

        print(f"\nIter {iteration}: ", end='', flush=True)

        # 파라미터 추출
        x = torch.tensor([
            params['edgeThresh1'],
            params['simThresh1'],
            params['pixelRatio1'],
            params['edgeThresh2'],
            params['simThresh2'],
            params['pixelRatio2'],
            params['ransac_weight_q'],
            params['ransac_weight_qg']
        ])

        # 해당 이미지에서 검출
        img_idx = iter_data['image_idx']
        img_data = images_data[img_idx]
        img_name = img_data['image_name']

        if img_name not in gt_data:
            print(f"GT 없음, skip")
            continue

        gt_coords = gt_data[img_name]

        # 검출 수행
        try:
            detected_coords = detect_lines_with_airline(
                image_path=img_data['image_path'],
                rois=img_data['rois'],
                x=x
            )
        except Exception as e:
            print(f"검출 실패: {e}")
            continue

        # 각 metric 계산
        m_current = line_equation_evaluation(detected_coords, gt_coords)
        m_exp_50 = metric_exponential(detected_coords, gt_coords, distance_scale=50.0)
        m_exp_100 = metric_exponential(detected_coords, gt_coords, distance_scale=100.0)
        m_exp_200 = metric_exponential(detected_coords, gt_coords, distance_scale=200.0)
        m_angle = metric_angle_based(detected_coords, gt_coords, distance_scale=100.0)
        m_end_30 = metric_endpoint_distance(detected_coords, gt_coords, distance_scale=30.0)
        m_end_50 = metric_endpoint_distance(detected_coords, gt_coords, distance_scale=50.0)
        m_end_100 = metric_endpoint_distance(detected_coords, gt_coords, distance_scale=100.0)
        m_weighted = metric_weighted(detected_coords, gt_coords, distance_scale=100.0)

        results['iteration'].append(iteration)
        results['score_original'].append(score_orig)
        results['cvar_original'].append(cvar_orig)
        results['metric_current'].append(m_current)
        results['metric_exponential_50'].append(m_exp_50)
        results['metric_exponential_100'].append(m_exp_100)
        results['metric_exponential_200'].append(m_exp_200)
        results['metric_angle_100'].append(m_angle)
        results['metric_endpoint_30'].append(m_end_30)
        results['metric_endpoint_50'].append(m_end_50)
        results['metric_endpoint_100'].append(m_end_100)
        results['metric_weighted'].append(m_weighted)

        print(f"Original={score_orig:.3f} | "
              f"Current={m_current:.3f} | "
              f"Exp(100)={m_exp_100:.3f} | "
              f"Angle={m_angle:.3f} | "
              f"Endpoint(50)={m_end_50:.3f} | "
              f"Weighted={m_weighted:.3f}")

    # 통계 및 시각화
    print("\n" + "="*70)
    print("통계 요약")
    print("="*70)

    for key in results.keys():
        if key in ['iteration']:
            continue
        vals = np.array(results[key])
        print(f"\n{key}:")
        print(f"  Mean: {vals.mean():.4f}  |  Std: {vals.std():.4f}")
        print(f"  Min:  {vals.min():.4f}  |  Max: {vals.max():.4f}")
        print(f"  Range: {vals.max() - vals.min():.4f}")

    # 시각화
    print("\n시각화 생성...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metric Comparison (Selected Iterations)', fontsize=16)

    # 1. Bar plot
    ax = axes[0, 0]
    iterations = results['iteration']
    x_pos = np.arange(len(iterations))
    width = 0.1

    metrics_to_plot = {
        'Original': results['score_original'],
        'Current': results['metric_current'],
        'Exp(50)': results['metric_exponential_50'],
        'Exp(100)': results['metric_exponential_100'],
        'Angle': results['metric_angle_100'],
        'Endpoint': results['metric_endpoint_50'],
        'Weighted': results['metric_weighted']
    }

    for i, (name, values) in enumerate(metrics_to_plot.items()):
        ax.bar(x_pos + i*width, values, width, label=name, alpha=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Score Comparison by Metric')
    ax.set_xticks(x_pos + width * 3)
    ax.set_xticklabels([f"#{i}" for i in iterations], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Correlation heatmap
    ax = axes[0, 1]
    metric_names = list(metrics_to_plot.keys())
    corr_matrix = np.zeros((len(metric_names), len(metric_names)))

    for i, name1 in enumerate(metric_names):
        for j, name2 in enumerate(metric_names):
            corr_matrix[i, j] = np.corrcoef(metrics_to_plot[name1],
                                            metrics_to_plot[name2])[0, 1]

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_yticklabels(metric_names)

    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Correlation Matrix')
    plt.colorbar(im, ax=ax)

    # 3. Distribution comparison
    ax = axes[1, 0]
    bp = ax.boxplot([v for v in metrics_to_plot.values()],
                    labels=list(metrics_to_plot.keys()),
                    patch_artist=True)
    ax.set_ylabel('Score')
    ax.set_title('Score Distributions')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Scatter: Original vs Best New Metric
    ax = axes[1, 1]

    # Find best correlation
    best_corr = 0
    best_metric_name = None
    for name, values in metrics_to_plot.items():
        if name == 'Original':
            continue
        corr = np.corrcoef(results['score_original'], values)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_metric_name = name

    if best_metric_name:
        ax.scatter(results['score_original'], metrics_to_plot[best_metric_name],
                  s=100, alpha=0.6, c=results['cvar_original'], cmap='viridis')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Original Score')
        ax.set_ylabel(f'{best_metric_name} Score')
        ax.set_title(f'Original vs {best_metric_name} (r={best_corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate points
        for i, iter_num in enumerate(iterations):
            ax.annotate(f'#{iter_num}',
                       (results['score_original'][i], metrics_to_plot[best_metric_name][i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()

    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / "metric_comparison_lite.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n시각화 저장: {fig_path}")

    json_path = output_path / "metric_comparison_lite.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"결과 저장: {json_path}")

    plt.show()

    print("\n" + "="*70)
    print("분석 완료!")
    print("="*70)

    return results


if __name__ == "__main__":
    print("Metric 비교 분석 시작...\n")

    # 데이터 로드
    log_dir = "logs/run_20251114_172045"
    output_dir = "results/metric_analysis"

    selected = select_representative_iterations(log_dir, n_samples=10)

    # 이미지 데이터 로드
    from optimization import load_dataset
    images_data = load_dataset(
        image_dir="../dataset/images/test",
        gt_file="../dataset/ground_truth_auto.json",
        complete_only=False
    )

    # 분석
    results = analyze_selected_iterations(selected, images_data, output_dir)

    print("\n\n" + "="*70)
    print("💡 권장사항")
    print("="*70)
    print("\n1. 위 그래프를 확인하세요:")
    print("   - Original Score와 가장 correlation 높은 metric")
    print("   - 분포가 가장 넓은 metric (변별력)")
    print("   - 시각적 품질과 일치하는 metric")
    print("\n2. distance_scale 파라미터:")
    print("   - Exp(50):  엄격 (40px = 0.45점)")
    print("   - Exp(100): 중간 (40px = 0.67점)")
    print("   - Exp(200): 관대 (40px = 0.82점)")
    print("\n3. 다음 단계:")
    print("   - 선택한 metric으로 optimization.py 수정")
    print("   - 새로운 BO 실험 시작")
