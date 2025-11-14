"""
실험 결과 기반 검출 Coverage 분석

Session 13 (run_20251114_172045) 결과를 분석:
1. iteration 파일에서 사용된 이미지 확인
2. 각 iteration에서 좌표 검출 성공/실패 확인
3. Upper point 검출 방식 분석 (교점 vs Fallback)
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
from collections import defaultdict


COORD_KEYS = [
    'longi_left_lower_x', 'longi_left_lower_y',
    'longi_right_lower_x', 'longi_right_lower_y',
    'longi_left_upper_x', 'longi_left_upper_y',
    'longi_right_upper_x', 'longi_right_upper_y',
    'collar_left_lower_x', 'collar_left_lower_y',
    'collar_left_upper_x', 'collar_left_upper_y',
]

COORD_GROUPS = {
    'longi_left_lower': ['longi_left_lower_x', 'longi_left_lower_y'],
    'longi_right_lower': ['longi_right_lower_x', 'longi_right_lower_y'],
    'longi_left_upper': ['longi_left_upper_x', 'longi_left_upper_y'],
    'longi_right_upper': ['longi_right_upper_x', 'longi_right_upper_y'],
    'collar_left_lower': ['collar_left_lower_x', 'collar_left_lower_y'],
    'collar_left_upper': ['collar_left_upper_x', 'collar_left_upper_y'],
}


def load_iteration_results(log_dir: str):
    """Iteration 결과 로드"""
    log_path = Path(log_dir)
    iter_files = sorted(log_path.glob("iter_*.json"))

    results = []
    for f in iter_files:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)

    return results


def check_point_detected(coords: Dict, point_name: str) -> bool:
    """특정 점이 검출되었는지 확인 (0이 아닌지)"""
    keys = COORD_GROUPS[point_name]
    x_val = coords.get(keys[0], 0)
    y_val = coords.get(keys[1], 0)
    return x_val != 0 and y_val != 0


def analyze_experiment_coverage(log_dir: str):
    """실험 결과 기반 Coverage 분석"""

    print("="*70)
    print("실험 결과 기반 검출 Coverage 분석")
    print(f"Log Dir: {log_dir}")
    print("="*70)

    # Iteration 결과 로드
    iter_results = load_iteration_results(log_dir)
    print(f"\n총 iterations: {len(iter_results)}")

    if len(iter_results) == 0:
        print("❌ Iteration 파일이 없습니다!")
        return

    # 체크할 키가 있는지 확인
    sample = iter_results[0]
    if 'detected_coords' not in sample:
        print("\n[!] WARNING: iteration 파일에 'detected_coords'가 없습니다!")
        print("현재 저장된 정보:")
        print(f"  - {', '.join(sample.keys())}")
        print("\n이 정보로는 검출 성공률을 확인할 수 없습니다.")
        print("대신 score 분포만 분석하겠습니다...\n")

        # Score만 분석
        scores = [r['score'] for r in iter_results]
        cvars = [r['cvar'] for r in iter_results]

        print("Score 통계:")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std:  {np.std(scores):.4f}")
        print(f"  Min:  {np.min(scores):.4f}")
        print(f"  Max:  {np.max(scores):.4f}")

        print("\nCVaR 통계:")
        print(f"  Mean: {np.mean(cvars):.4f}")
        print(f"  Std:  {np.std(cvars):.4f}")
        print(f"  Min:  {np.min(cvars):.4f}")
        print(f"  Max:  {np.max(cvars):.4f}")

        # Score가 낮은 케이스 (검출 실패 의심)
        low_score_iters = [(i, r['score'], r.get('image_idx', '?'))
                          for i, r in enumerate(iter_results) if r['score'] < 0.3]

        print(f"\nScore < 0.3 (detection failure suspected): {len(low_score_iters)}개")
        if low_score_iters:
            print("  Top 10:")
            for iter_num, score, img_idx in low_score_iters[:10]:
                print(f"    Iter {iter_num+1}: score={score:.4f}, image_idx={img_idx}")

        # Score가 높은 케이스
        high_score_iters = [(i, r['score'], r.get('image_idx', '?'))
                           for i, r in enumerate(iter_results) if r['score'] > 0.7]

        print(f"\nScore > 0.7 (detection success): {len(high_score_iters)}개")
        if high_score_iters:
            print("  상위 10개:")
            for iter_num, score, img_idx in high_score_iters[:10]:
                print(f"    Iter {iter_num+1}: score={score:.4f}, image_idx={img_idx}")

        # 사용된 이미지 분석
        image_indices = [r.get('image_idx', -1) for r in iter_results if 'image_idx' in r]
        if image_indices:
            unique_images = len(set(image_indices))
            print(f"\n사용된 고유 이미지 수: {unique_images}")
            print(f"전체 평가 횟수: {len(image_indices)}")
            print(f"평균 재사용: {len(image_indices) / unique_images:.1f}회")

        return

    # detected_coords가 있는 경우 (원래 계획대로)
    print("\n✓ detected_coords 발견! 상세 분석 시작...\n")

    # 결과 저장
    coverage_stats = {
        'total': len(iter_results),
        'point_success': {name: 0 for name in COORD_GROUPS.keys()},
        'point_failure': {name: [] for name in COORD_GROUPS.keys()},
        'all_success': 0,
        'all_failure': [],
        'per_iteration': []
    }

    # 각 iteration 분석
    for i, iter_data in enumerate(iter_results):
        iteration = iter_data['iteration']
        detected_coords = iter_data['detected_coords']
        score = iter_data.get('score', 0)
        image_idx = iter_data.get('image_idx', '?')

        # 각 점 검출 여부
        point_status = {}
        all_detected = True

        for point_name in COORD_GROUPS.keys():
            detected = check_point_detected(detected_coords, point_name)
            point_status[point_name] = detected

            if detected:
                coverage_stats['point_success'][point_name] += 1
            else:
                coverage_stats['point_failure'][point_name].append(iteration)
                all_detected = False

        if all_detected:
            coverage_stats['all_success'] += 1
        else:
            coverage_stats['all_failure'].append(iteration)

        coverage_stats['per_iteration'].append({
            'iteration': iteration,
            'image_idx': image_idx,
            'score': score,
            'status': point_status,
            'all_detected': all_detected
        })

    # 통계 출력
    print("="*70)
    print("검출 성공률 요약")
    print("="*70)

    print("\n각 점별 검출 성공률:")
    for point_name, success_count in coverage_stats['point_success'].items():
        failure_count = len(coverage_stats['point_failure'][point_name])
        success_rate = (success_count / coverage_stats['total']) * 100
        status = "✓" if success_rate >= 90 else "⚠" if success_rate >= 70 else "✗"
        print(f"  {status} {point_name:20s}: {success_count:3d}/{coverage_stats['total']:3d} "
              f"({success_rate:5.1f}%)")

    all_success_rate = (coverage_stats['all_success'] / coverage_stats['total']) * 100
    print(f"\n모든 점 검출 성공: {coverage_stats['all_success']}/{coverage_stats['total']} "
          f"({all_success_rate:.1f}%)")

    # Upper point 특별 분석
    print("\n" + "="*70)
    print("Upper Point 검출 분석 (교점 방식의 핵심)")
    print("="*70)

    upper_points = ['longi_left_upper', 'longi_right_upper']
    for pt in upper_points:
        success = coverage_stats['point_success'][pt]
        failure = len(coverage_stats['point_failure'][pt])
        rate = (success / coverage_stats['total']) * 100

        print(f"\n{pt}:")
        print(f"  성공: {success}/{coverage_stats['total']} ({rate:.1f}%)")
        print(f"  실패: {failure}개 iterations")

        if failure > 0 and failure <= 10:
            failed_iters = coverage_stats['point_failure'][pt]
            print(f"  실패 iterations: {failed_iters}")

    # Collar 분석
    print("\n" + "="*70)
    print("Collar Point 검출 분석")
    print("="*70)

    collar_points = ['collar_left_lower', 'collar_left_upper']
    for pt in collar_points:
        success = coverage_stats['point_success'][pt]
        failure = len(coverage_stats['point_failure'][pt])
        rate = (success / coverage_stats['total']) * 100

        print(f"\n{pt}:")
        print(f"  성공: {success}/{coverage_stats['total']} ({rate:.1f}%)")
        print(f"  실패: {failure}개 iterations")

    # 시각화
    print("\n시각화 생성 중...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Experiment Coverage Analysis\n{log_dir}', fontsize=14)

    # 1. 점별 성공률
    ax = axes[0, 0]
    point_names = list(coverage_stats['point_success'].keys())
    success_rates = [(coverage_stats['point_success'][p] / coverage_stats['total']) * 100
                     for p in point_names]
    colors = ['green' if r >= 90 else 'orange' if r >= 70 else 'red'
              for r in success_rates]

    ax.barh(point_names, success_rates, color=colors, alpha=0.7)
    ax.set_xlabel('Success Rate (%)')
    ax.set_title('Detection Success Rate by Point')
    ax.axvline(90, color='green', linestyle='--', alpha=0.3)
    ax.axvline(70, color='orange', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Iteration별 검출 성공 추이
    ax = axes[0, 1]
    iterations = [d['iteration'] for d in coverage_stats['per_iteration']]
    n_failed = [sum(1 for v in d['status'].values() if not v)
                for d in coverage_stats['per_iteration']]

    ax.plot(iterations, n_failed, 'o-', alpha=0.6, markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Failed Points')
    ax.set_title('Detection Failures Over Time')
    ax.grid(True, alpha=0.3)

    # 3. Score vs 검출 성공 개수
    ax = axes[1, 0]
    scores = [d['score'] for d in coverage_stats['per_iteration']]
    n_success = [sum(1 for v in d['status'].values() if v)
                 for d in coverage_stats['per_iteration']]

    scatter = ax.scatter(scores, n_success, c=iterations, cmap='viridis',
                        alpha=0.6, s=50)
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of Detected Points')
    ax.set_title('Score vs Detection Success')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Iteration')

    # 4. 실패 패턴 분석
    ax = axes[1, 1]

    # 가장 많이 실패한 점
    failure_counts = {name: len(failures)
                     for name, failures in coverage_stats['point_failure'].items()}
    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

    names = [item[0] for item in sorted_failures]
    counts = [item[1] for item in sorted_failures]

    ax.barh(names, counts, alpha=0.7, color='salmon')
    ax.set_xlabel('Number of Failures')
    ax.set_title('Most Common Detection Failures')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # 저장
    output_dir = Path(log_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "experiment_coverage.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"시각화 저장: {fig_path}")

    json_path = output_dir / "experiment_coverage.json"
    with open(json_path, 'w') as f:
        json.dump(coverage_stats, f, indent=2)
    print(f"결과 저장: {json_path}")

    plt.show()

    # 최종 평가
    print("\n" + "="*70)
    print("최종 평가")
    print("="*70)

    upper_min_rate = min(
        coverage_stats['point_success']['longi_left_upper'] / coverage_stats['total'] * 100,
        coverage_stats['point_success']['longi_right_upper'] / coverage_stats['total'] * 100
    )

    if upper_min_rate >= 95:
        print("✓ Upper Point 검출 매우 우수 (95%+)")
        print("  → 교점 방식이 안정적으로 작동")
    elif upper_min_rate >= 85:
        print("✓ Upper Point 검출 우수 (85%+)")
        print("  → 대부분 교점으로 검출 성공")
    elif upper_min_rate >= 70:
        print("⚠ Upper Point 검출 보통 (70-85%)")
        print("  → 일부 실패, Fallback 비율 확인 필요")
    else:
        print("✗ Upper Point 검출 불안정 (<70%)")
        print("  → 교점 방식의 한계 확인됨")
        print("  → Fallback이 주로 사용되고 있을 가능성")

    print(f"\n전체 검출 성공률: {all_success_rate:.1f}%")

    if all_success_rate >= 80:
        print("  → 대부분의 iteration에서 완전 검출 성공")
    elif all_success_rate >= 50:
        print("  → 절반 정도만 완전 검출")
        print("  → Metric 설계 시 부분 검출 고려 필요")
    else:
        print("  → 대부분 일부 점 실패")
        print("  → 심각한 검출 문제 또는 Collar 누락")

    return coverage_stats


if __name__ == "__main__":
    log_dir = "logs/run_20251114_172045"

    print("Session 13 실험 결과 분석")
    print(f"대상: {log_dir}\n")

    results = analyze_experiment_coverage(log_dir)
