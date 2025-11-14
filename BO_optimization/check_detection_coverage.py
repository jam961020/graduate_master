"""
검출 Coverage 분석

전체 데이터셋(113개 이미지)에 대해:
1. Default 파라미터로 검출
2. 각 좌표(12개) 검출 성공/실패 확인
3. 실패 케이스 분석 및 시각화
"""

import json
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List
import matplotlib.pyplot as plt
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector
import torch

# Default 파라미터 (초기값)
DEFAULT_PARAMS = {
    'edgeThresh1': -3.0,
    'simThresh1': 0.98,
    'pixelRatio1': 0.05,
    'edgeThresh2': 1.0,
    'simThresh2': 0.75,
    'pixelRatio2': 0.05,
}

COORD_KEYS = [
    'longi_left_lower_x', 'longi_left_lower_y',
    'longi_right_lower_x', 'longi_right_lower_y',
    'longi_left_upper_x', 'longi_left_upper_y',
    'longi_right_upper_x', 'longi_right_upper_y',
    'collar_left_lower_x', 'collar_left_lower_y',
    'collar_left_upper_x', 'collar_left_upper_y',
]

# 그룹별로 묶기
COORD_GROUPS = {
    'longi_left_lower': ['longi_left_lower_x', 'longi_left_lower_y'],
    'longi_right_lower': ['longi_right_lower_x', 'longi_right_lower_y'],
    'longi_left_upper': ['longi_left_upper_x', 'longi_left_upper_y'],
    'longi_right_upper': ['longi_right_upper_x', 'longi_right_upper_y'],
    'collar_left_lower': ['collar_left_lower_x', 'collar_left_lower_y'],
    'collar_left_upper': ['collar_left_upper_x', 'collar_left_upper_y'],
}


def load_images_and_gt(image_dir: str, gt_file: str):
    """이미지와 GT 로드"""
    image_dir = Path(image_dir)

    with open(gt_file) as f:
        gt_data = json.load(f)

    images = []
    for img_name in sorted(gt_data.keys()):
        # GT 키에 확장자가 없으면 추가
        possible_paths = [
            image_dir / f"{img_name}.jpg",
            image_dir / f"{img_name}.png",
            image_dir / img_name,  # 이미 확장자 있는 경우
        ]

        img_path = None
        for p in possible_paths:
            if p.exists():
                img_path = p
                break

        if img_path:
            gt_coords = gt_data[img_name].get('coordinates', gt_data[img_name])
            images.append({
                'name': img_name,
                'path': str(img_path),
                'gt': gt_coords
            })

    return images


def check_point_detected(coords: Dict, point_name: str) -> bool:
    """특정 점이 검출되었는지 확인 (0이 아닌지)"""
    keys = COORD_GROUPS[point_name]
    x_val = coords.get(keys[0], 0)
    y_val = coords.get(keys[1], 0)
    return x_val != 0 and y_val != 0


def analyze_detection_coverage(image_dir: str, gt_file: str, yolo_model_path: str):
    """전체 데이터셋 검출 Coverage 분석"""

    print("="*70)
    print("검출 Coverage 분석")
    print("="*70)

    # 데이터 로드
    images = load_images_and_gt(image_dir, gt_file)
    print(f"\n총 이미지 수: {len(images)}")

    # YOLO 로드
    print("YOLO 모델 로딩...")
    yolo_detector = YOLODetector(yolo_model_path)

    # 결과 저장
    results = {
        'total': len(images),
        'point_success': {name: 0 for name in COORD_GROUPS.keys()},
        'point_failure': {name: [] for name in COORD_GROUPS.keys()},
        'all_success': 0,
        'all_failure': [],
        'detection_results': []
    }

    print("\n검출 진행 중...")
    for i, img_data in enumerate(images):
        print(f"\r[{i+1}/{len(images)}] {img_data['name']}", end='', flush=True)

        # 이미지 로드
        image = cv2.imread(img_data['path'])
        if image is None:
            continue

        # 검출
        try:
            detected_coords = detect_with_full_pipeline(
                image, DEFAULT_PARAMS, yolo_detector
            )
        except Exception as e:
            print(f"\n  ✗ 검출 실패: {e}")
            detected_coords = {k: 0 for k in COORD_KEYS}

        # 각 점 검출 여부 확인
        point_status = {}
        all_detected = True

        for point_name in COORD_GROUPS.keys():
            detected = check_point_detected(detected_coords, point_name)
            point_status[point_name] = detected

            if detected:
                results['point_success'][point_name] += 1
            else:
                results['point_failure'][point_name].append(img_data['name'])
                all_detected = False

        # 전체 검출 성공 여부
        if all_detected:
            results['all_success'] += 1
        else:
            results['all_failure'].append(img_data['name'])

        results['detection_results'].append({
            'image': img_data['name'],
            'status': point_status,
            'all_detected': all_detected
        })

    print("\n\n" + "="*70)
    print("검출 성공률 요약")
    print("="*70)

    # 점별 성공률
    print("\n각 점별 검출 성공률:")
    for point_name, success_count in results['point_success'].items():
        failure_count = len(results['point_failure'][point_name])
        success_rate = (success_count / results['total']) * 100
        print(f"  {point_name:20s}: {success_count:3d}/{results['total']:3d} "
              f"({success_rate:5.1f}%)  실패: {failure_count}개")

    # 전체 검출 성공률
    all_success_rate = (results['all_success'] / results['total']) * 100
    print(f"\n모든 점 검출 성공: {results['all_success']}/{results['total']} ({all_success_rate:.1f}%)")
    print(f"일부 실패: {len(results['all_failure'])}개")

    # 실패 케이스 분석
    print("\n" + "="*70)
    print("실패 패턴 분석")
    print("="*70)

    # Upper point 실패 (가장 중요!)
    upper_points = ['longi_left_upper', 'longi_right_upper']
    upper_failures = set()
    for pt in upper_points:
        upper_failures.update(results['point_failure'][pt])

    print(f"\nUpper Point 실패: {len(upper_failures)}개 이미지")
    if len(upper_failures) > 0:
        print("  실패 이미지:")
        for img_name in sorted(upper_failures)[:10]:  # 최대 10개만
            print(f"    - {img_name}")
        if len(upper_failures) > 10:
            print(f"    ... 외 {len(upper_failures)-10}개")

    # Collar 실패
    collar_points = ['collar_left_lower', 'collar_left_upper']
    collar_failures = set()
    for pt in collar_points:
        collar_failures.update(results['point_failure'][pt])

    print(f"\nCollar Point 실패: {len(collar_failures)}개 이미지")
    if len(collar_failures) > 0:
        print("  실패 이미지:")
        for img_name in sorted(collar_failures)[:10]:
            print(f"    - {img_name}")
        if len(collar_failures) > 10:
            print(f"    ... 외 {len(collar_failures)-10}개")

    # 시각화
    print("\n시각화 생성 중...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detection Coverage Analysis', fontsize=16)

    # 1. 점별 성공률 Bar chart
    ax = axes[0, 0]
    point_names = list(results['point_success'].keys())
    success_rates = [(results['point_success'][p] / results['total']) * 100
                     for p in point_names]
    colors = ['green' if r >= 90 else 'orange' if r >= 70 else 'red'
              for r in success_rates]

    ax.barh(point_names, success_rates, color=colors, alpha=0.7)
    ax.set_xlabel('Success Rate (%)')
    ax.set_title('Detection Success Rate by Point')
    ax.axvline(90, color='green', linestyle='--', alpha=0.3, label='90%')
    ax.axvline(70, color='orange', linestyle='--', alpha=0.3, label='70%')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    # 2. 전체 성공/실패 Pie chart
    ax = axes[0, 1]
    sizes = [results['all_success'], len(results['all_failure'])]
    labels = [f'All Detected\n({results["all_success"]})',
              f'Partial Failure\n({len(results["all_failure"])})']
    colors_pie = ['#90EE90', '#FFB6C1']
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title('Overall Detection Success')

    # 3. 실패 개수 분포 (이미지별로 몇 개 점이 실패했는지)
    ax = axes[1, 0]
    failure_counts = []
    for res in results['detection_results']:
        n_failures = sum(1 for v in res['status'].values() if not v)
        failure_counts.append(n_failures)

    bins = range(0, 7)
    ax.hist(failure_counts, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Failed Points per Image')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Failures')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Upper vs Lower 성공률 비교
    ax = axes[1, 1]
    categories = ['Lower Left', 'Lower Right', 'Upper Left', 'Upper Right',
                  'Collar Lower', 'Collar Upper']
    success = [
        results['point_success']['longi_left_lower'] / results['total'] * 100,
        results['point_success']['longi_right_lower'] / results['total'] * 100,
        results['point_success']['longi_left_upper'] / results['total'] * 100,
        results['point_success']['longi_right_upper'] / results['total'] * 100,
        results['point_success']['collar_left_lower'] / results['total'] * 100,
        results['point_success']['collar_left_upper'] / results['total'] * 100,
    ]

    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, success, alpha=0.7)
    bars[0].set_color('blue')
    bars[1].set_color('blue')
    bars[2].set_color('red')
    bars[3].set_color('red')
    bars[4].set_color('green')
    bars[5].set_color('green')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Point Type')
    ax.axhline(90, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 저장
    output_dir = Path("results/coverage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "detection_coverage.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"시각화 저장: {fig_path}")

    # JSON 저장
    json_path = output_dir / "detection_coverage.json"
    # 저장 전 처리 (set을 list로 변환)
    results_serializable = {
        'total': results['total'],
        'point_success': results['point_success'],
        'point_failure': {k: list(v) for k, v in results['point_failure'].items()},
        'all_success': results['all_success'],
        'all_failure': results['all_failure'],
        'detection_results': results['detection_results']
    }
    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"결과 저장: {json_path}")

    plt.show()

    # 최종 평가
    print("\n" + "="*70)
    print("최종 평가")
    print("="*70)

    if all_success_rate >= 95:
        print("✓ 매우 우수: 95% 이상 완전 검출")
    elif all_success_rate >= 85:
        print("✓ 우수: 85% 이상 완전 검출")
    elif all_success_rate >= 70:
        print("⚠ 보통: 70~85% 완전 검출 (개선 필요)")
    else:
        print("✗ 나쁨: 70% 미만 완전 검출 (심각한 문제)")

    # Upper point 특별 체크
    upper_success_rate = min(
        results['point_success']['longi_left_upper'] / results['total'] * 100,
        results['point_success']['longi_right_upper'] / results['total'] * 100
    )

    print(f"\nUpper Point 최소 성공률: {upper_success_rate:.1f}%")
    if upper_success_rate < 80:
        print("  ⚠ 경고: Upper Point 검출이 불안정합니다!")
        print("  → 교점 방식의 한계 확인됨")
        print("  → Fallback 비율 확인 필요")

    return results


if __name__ == "__main__":
    image_dir = "../dataset/images/test"
    gt_file = "../dataset/ground_truth_auto.json"
    yolo_model = "models/best.pt"

    results = analyze_detection_coverage(image_dir, gt_file, yolo_model)

    print("\n\n" + "="*70)
    print("💡 권장사항")
    print("="*70)

    # Upper point 실패율이 높으면
    upper_failure_rate = (len(results['point_failure']['longi_left_upper']) / results['total']) * 100
    if upper_failure_rate > 20:
        print("\n1. Upper Point 검출 불안정 (>20% 실패)")
        print("   - 교점 방식의 한계")
        print("   - Fallback(ROI 경계) 비율 높음")
        print("   - 대안: ROI 경계 + offset을 기본으로?")

    # Collar 실패율이 높으면
    collar_failure_rate = (len(results['point_failure']['collar_left_lower']) / results['total']) * 100
    if collar_failure_rate > 30:
        print("\n2. Collar 검출 불안정 (>30% 실패)")
        print("   - Collar plate가 없는 이미지 많음")
        print("   - 또는 검출 실패")
        print("   - Metric에서 Collar 가중치 낮추기")

    # 전체 성공률이 낮으면
    if results['all_success'] / results['total'] < 0.7:
        print("\n3. 전체 검출 성공률 낮음 (<70%)")
        print("   - 파라미터 튜닝 필요")
        print("   - 또는 데이터셋 품질 문제")
