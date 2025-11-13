#!/usr/bin/env python3
"""
실험 결과 분석 스크립트
- 환경 파라미터 유의미성 분석
- alpha 영향도 분석
- CVaR vs Mean 비교
"""

import json
import numpy as np
import os
from pathlib import Path
from scipy import stats
from collections import defaultdict

def load_all_results(results_dir="results"):
    """모든 실험 결과 로드"""
    results = []
    results_path = Path(results_dir)

    for json_file in sorted(results_path.glob("bo_cvar_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['filename'] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results

def analyze_environment_impact(results):
    """환경 파라미터의 성능 영향도 분석"""
    print("\n" + "="*60)
    print("환경 파라미터 영향도 분석")
    print("="*60)

    for result in results:
        alpha = result.get('alpha', 'unknown')
        n_w = result.get('n_w', 'unknown')
        final_cvar = result.get('final_cvar', 0)
        iterations = result.get('iterations', 0)

        print(f"\n실험: {result['filename']}")
        print(f"  alpha={alpha}, n_w={n_w}")
        print(f"  Final CVaR: {final_cvar:.4f}")
        print(f"  Iterations: {iterations}")

        # Best parameters 분석
        if 'best_params' in result:
            best_params = result['best_params']
            print(f"  Best parameters:")
            for param, value in best_params.items():
                print(f"    {param}: {value:.4f}")

def analyze_alpha_effect(results):
    """Alpha 값에 따른 성능 변화 분석"""
    print("\n" + "="*60)
    print("Alpha 영향도 분석")
    print("="*60)

    # Alpha별로 그룹화
    alpha_groups = defaultdict(list)
    for result in results:
        alpha = result.get('alpha', None)
        if alpha is not None:
            final_cvar = result.get('final_cvar', 0)
            alpha_groups[alpha].append(final_cvar)

    # 정렬 및 출력
    for alpha in sorted(alpha_groups.keys()):
        cvars = alpha_groups[alpha]
        print(f"\nalpha={alpha}:")
        print(f"  실험 횟수: {len(cvars)}")
        print(f"  평균 CVaR: {np.mean(cvars):.4f}")
        print(f"  최고 CVaR: {np.max(cvars):.4f}")
        print(f"  최저 CVaR: {np.min(cvars):.4f}")
        print(f"  표준편차: {np.std(cvars):.4f}")

def analyze_convergence(results):
    """수렴 속도 분석"""
    print("\n" + "="*60)
    print("수렴 속도 분석")
    print("="*60)

    for result in results:
        history = result.get('history', {})
        if not history:
            continue

        # history가 dict이 아닌 경우 (list 등) 스킵
        if not isinstance(history, dict):
            continue

        iterations = history.get('iterations', [])
        cvars = history.get('cvars', [])

        if len(cvars) < 2:
            continue

        print(f"\n실험: {result['filename']}")
        print(f"  초기 CVaR: {cvars[0]:.4f}")
        print(f"  최종 CVaR: {cvars[-1]:.4f}")
        print(f"  개선도: {(cvars[-1] - cvars[0]):.4f} ({((cvars[-1]/cvars[0] - 1)*100):.2f}%)")

        # 개선이 멈춘 지점 찾기
        improvement_threshold = 0.01
        last_improvement_iter = 0
        for i in range(1, len(cvars)):
            if cvars[i] - cvars[i-1] > improvement_threshold:
                last_improvement_iter = iterations[i]

        print(f"  마지막 유의미한 개선: iteration {last_improvement_iter}")

def analyze_environment_correlation():
    """환경 파라미터 간 상관관계 분석"""
    print("\n" + "="*60)
    print("환경 파라미터 상관관계 분석")
    print("="*60)

    # 이 부분은 environment features 데이터가 필요
    # 로그 파일에서 추출 필요
    print("  [TODO] 로그 파일에서 환경 특징 추출 후 분석")

def generate_summary_report(results):
    """종합 요약 리포트"""
    print("\n" + "="*60)
    print("종합 요약 리포트")
    print("="*60)

    total_experiments = len(results)
    print(f"\n전체 실험 수: {total_experiments}")

    # 최고 성능 찾기
    best_result = max(results, key=lambda x: x.get('final_cvar', 0))
    print(f"\n최고 성능:")
    print(f"  파일: {best_result['filename']}")
    print(f"  CVaR: {best_result.get('final_cvar', 0):.4f}")
    print(f"  alpha: {best_result.get('alpha', 'unknown')}")
    print(f"  n_w: {best_result.get('n_w', 'unknown')}")

    # Alpha별 통계
    alphas = [r.get('alpha') for r in results if r.get('alpha') is not None]
    if alphas:
        print(f"\nAlpha 값 범위: {min(alphas):.2f} ~ {max(alphas):.2f}")
        print(f"테스트된 Alpha 값: {sorted(set(alphas))}")

def main():
    print("="*60)
    print("BoRisk 실험 결과 분석")
    print("="*60)

    # 결과 로드
    results = load_all_results()

    if not results:
        print("\n경고: 분석할 결과 파일이 없습니다!")
        return

    print(f"\n로드된 결과: {len(results)}개")

    # 분석 실행
    analyze_environment_impact(results)
    analyze_alpha_effect(results)
    analyze_convergence(results)
    analyze_environment_correlation()
    generate_summary_report(results)

    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
