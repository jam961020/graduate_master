"""
Session 13 결과로 Metric 비교
3가지 metric으로 대표 케이스 재평가
"""
import json
import numpy as np
import torch
from pathlib import Path
import cv2
import sys

# 현재 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from optimization import line_equation_evaluation, detect_with_full_pipeline, BOUNDS, DEVICE, DTYPE
from evaluation import evaluate_lp, evaluate_endpoint_error, GT_LABELS
from yolo_detector import YOLODetector

def load_iteration_data(log_dir, iteration):
    """iteration JSON에서 파라미터와 이미지 정보 로드"""
    iter_file = log_dir / f'iter_{iteration:03d}.json'

    with open(iter_file) as f:
        data = json.load(f)

    return data

def evaluate_with_all_metrics(detected_coords, gt_coords, image, image_name, image_size):
    """3가지 metric으로 평가"""
    h, w = image_size

    # 1. line_equation_evaluation (현재)
    score_line_eq = line_equation_evaluation(detected_coords, gt_coords, image_size=(w, h))

    # 2. lp (F1 score)
    score_lp = evaluate_lp(detected_coords, image, image_name, threshold=50.0, debug=False)

    # 3. endpoint
    score_endpoint = evaluate_endpoint_error(detected_coords, image, image_name, debug=False)

    return {
        'line_eq': score_line_eq,
        'lp': score_lp,
        'endpoint': score_endpoint
    }

def main():
    log_dir = Path('logs/run_20251114_172045')

    # 대표 케이스 (앞에서 분석한 결과)
    cases = [
        {'name': 'Best', 'iteration': 64, 'expected_score': 0.8329},
        {'name': 'Median', 'iteration': 115, 'expected_score': 0.4587},
        {'name': 'Worst', 'iteration': 28, 'expected_score': 0.2529},
    ]

    # YOLO 초기화
    print("Loading YOLO...")
    yolo_detector = YOLODetector(model_path="models/best.pt")

    # 데이터셋 로드
    print("Loading dataset...")
    from optimization import load_dataset
    images_data = load_dataset(
        image_dir="../dataset/images/test",
        gt_file="../dataset/ground_truth.json",
        complete_only=False
    )

    print(f"\n{'='*80}")
    print("METRIC COMPARISON - Session 13 Representative Cases")
    print(f"{'='*80}\n")

    results = []

    for case in cases:
        print(f"\n{case['name']} Case - Iteration {case['iteration']}")
        print("-" * 60)

        # Iteration 데이터 로드
        data = load_iteration_data(log_dir, case['iteration'])

        params_dict = data['parameters']
        image_idx = data['image_idx']
        current_score = data['score']

        print(f"  Image: {images_data[image_idx]['name']}")
        print(f"  Current score (line_eq): {current_score:.4f}")

        # 파라미터 변환 (딕셔너리로)
        params = {
            'edgeThresh1': params_dict['edgeThresh1'],
            'simThresh1': params_dict['simThresh1'],
            'pixelRatio1': params_dict['pixelRatio1'],
            'edgeThresh2': params_dict['edgeThresh2'],
            'simThresh2': params_dict['simThresh2'],
            'pixelRatio2': params_dict['pixelRatio2']
        }

        # RANSAC 가중치 (튜플로)
        ransac_weights = (
            params_dict['ransac_weight_q'],
            params_dict['ransac_weight_qg']
        )

        # 이미지 준비
        image = images_data[image_idx]['image']
        image_name = images_data[image_idx]['name']
        gt_coords = images_data[image_idx]['gt_coords']

        # 검출 수행
        detected_coords = detect_with_full_pipeline(image, params, yolo_detector, ransac_weights)

        # 3가지 metric으로 평가
        scores = evaluate_with_all_metrics(
            detected_coords,
            gt_coords,
            image,
            image_name,
            image.shape[:2]
        )

        print(f"\n  Metric Comparison:")
        print(f"    line_eq:  {scores['line_eq']:.4f} (current)")
        print(f"    lp:       {scores['lp']:.4f}")
        print(f"    endpoint: {scores['endpoint']:.4f}")

        # 결과 저장
        results.append({
            'case': case['name'],
            'iteration': case['iteration'],
            'image_name': image_name,
            'scores': scores
        })

    # 비교표 생성
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")

    print(f"{'Case':<10} {'Iter':<6} {'Image':<20} {'line_eq':<10} {'lp':<10} {'endpoint':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['case']:<10} {r['iteration']:<6} {r['image_name']:<20} "
              f"{r['scores']['line_eq']:<10.4f} {r['scores']['lp']:<10.4f} "
              f"{r['scores']['endpoint']:<10.4f}")

    # 상관관계 분석
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}\n")

    line_eq_scores = [r['scores']['line_eq'] for r in results]
    lp_scores = [r['scores']['lp'] for r in results]
    endpoint_scores = [r['scores']['endpoint'] for r in results]

    # line_eq vs lp
    corr_lp = np.corrcoef(line_eq_scores, lp_scores)[0, 1]
    print(f"line_eq vs lp:       {corr_lp:.4f}")

    # line_eq vs endpoint
    corr_endpoint = np.corrcoef(line_eq_scores, endpoint_scores)[0, 1]
    print(f"line_eq vs endpoint: {corr_endpoint:.4f}")

    # lp vs endpoint
    corr_lp_endpoint = np.corrcoef(lp_scores, endpoint_scores)[0, 1]
    print(f"lp vs endpoint:      {corr_lp_endpoint:.4f}")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
1. line_eq (현재 사용 중):
   - 직선 방정식 기반 (기울기 + 평행 거리)
   - threshold=40px (작음)
   - 문제: 1개 실패 시 급락

2. lp (F1 score):
   - Precision & Recall 기반
   - threshold=50px
   - 부분 점수 있음

3. endpoint:
   - 끝점 거리만 사용
   - 가장 직관적
   - 방향 정보 없음

→ Best metric은 시각적 품질과 가장 일치하는 것!
→ 3개 케이스 육안 확인 필요
""")

    # 결과 저장
    output_file = 'metric_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

if __name__ == '__main__':
    main()
