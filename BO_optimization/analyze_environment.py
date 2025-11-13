#!/usr/bin/env python3
"""
환경 파라미터 유의미성 분석
- 어떤 환경 특징이 성능에 영향?
- 환경 특징 간 상관관계
- w_set 샘플링 다양성 분석
"""

import json
import numpy as np
import torch
from pathlib import Path
import sys

# 프로젝트 모듈 import
try:
    from environment_independent import extract_environment_features
except ImportError:
    print("Warning: Could not import environment_independent")

def load_images_and_features(image_dir="../dataset/images/test", max_images=None):
    """이미지 로드 및 환경 특징 추출"""
    from PIL import Image
    import os

    print("Loading images and extracting environment features...")

    image_paths = sorted(Path(image_dir).glob("*.jpg"))[:max_images] if max_images else sorted(Path(image_dir).glob("*.jpg"))

    images = []
    features = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            # 환경 특징 추출
            env_features = extract_environment_features(img_np)

            images.append(str(img_path))
            features.append(env_features)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    features_tensor = torch.stack(features)
    return images, features_tensor

def analyze_feature_statistics(features):
    """환경 특징 통계"""
    print("\n" + "="*60)
    print("환경 특징 통계")
    print("="*60)

    feature_names = ['brightness', 'contrast', 'edge_density',
                     'texture_complexity', 'blur_level', 'noise_level']

    features_np = features.numpy()

    for i, name in enumerate(feature_names):
        values = features_np[:, i]
        print(f"\n{name}:")
        print(f"  평균: {np.mean(values):.4f}")
        print(f"  표준편차: {np.std(values):.4f}")
        print(f"  최소: {np.min(values):.4f}")
        print(f"  최대: {np.max(values):.4f}")
        print(f"  중앙값: {np.median(values):.4f}")

def analyze_feature_correlation(features):
    """환경 특징 간 상관관계"""
    print("\n" + "="*60)
    print("환경 특징 상관관계")
    print("="*60)

    feature_names = ['brightness', 'contrast', 'edge_density',
                     'texture_complexity', 'blur_level', 'noise_level']

    features_np = features.numpy()

    # 상관계수 행렬 계산
    corr_matrix = np.corrcoef(features_np.T)

    print("\n상관계수 행렬:")
    print("           ", end="")
    for name in feature_names:
        print(f"{name[:8]:>10}", end="")
    print()

    for i, name in enumerate(feature_names):
        print(f"{name[:10]:>12}", end="")
        for j in range(len(feature_names)):
            corr = corr_matrix[i, j]
            # 높은 상관관계는 강조
            if abs(corr) > 0.7 and i != j:
                print(f" {corr:>9.3f}*", end="")
            else:
                print(f" {corr:>9.3f} ", end="")
        print()

    # 높은 상관관계 찾기
    print("\n높은 상관관계 (|r| > 0.7):")
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.7:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: {corr:.3f}")

def analyze_sampling_diversity(features, n_samples=15):
    """w_set 샘플링 다양성 분석"""
    print("\n" + "="*60)
    print(f"샘플링 다양성 분석 (n_w={n_samples})")
    print("="*60)

    n_images = features.shape[0]

    # 10회 샘플링 시뮬레이션
    print("\n10회 샘플링 시뮬레이션:")
    diversities = []

    for trial in range(10):
        # 랜덤 샘플링
        indices = np.random.choice(n_images, size=n_samples, replace=False)
        sampled = features[indices].numpy()

        # 다양성 측정: 쌍별 거리의 평균
        dists = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(sampled[i] - sampled[j])
                dists.append(dist)

        avg_dist = np.mean(dists)
        diversities.append(avg_dist)

        print(f"  Trial {trial+1}: 평균 쌍별 거리 = {avg_dist:.4f}")

    print(f"\n다양성 통계:")
    print(f"  평균: {np.mean(diversities):.4f}")
    print(f"  표준편차: {np.std(diversities):.4f}")

    # n_w 크기별 다양성 비교
    print("\nn_w 크기별 다양성:")
    for n_w in [3, 5, 8, 10, 15, 20]:
        if n_w > n_images:
            continue
        indices = np.random.choice(n_images, size=n_w, replace=False)
        sampled = features[indices].numpy()

        dists = []
        for i in range(n_w):
            for j in range(i+1, n_w):
                dist = np.linalg.norm(sampled[i] - sampled[j])
                dists.append(dist)

        avg_dist = np.mean(dists)
        print(f"  n_w={n_w:2d}: {avg_dist:.4f}")

def analyze_performance_vs_environment():
    """성능과 환경 특징의 관계 (로그 파일 필요)"""
    print("\n" + "="*60)
    print("성능-환경 특징 관계 분석")
    print("="*60)

    print("  [TODO] 로그 파일에서:")
    print("  - 각 iteration의 선택된 w_idx")
    print("  - 해당 환경의 특징")
    print("  - 성능 점수")
    print("  → 어떤 환경에서 성능이 좋은지 분석")

def compare_w_set_strategies(features, n_trials=5):
    """다양한 w_set 샘플링 전략 비교"""
    print("\n" + "="*60)
    print("샘플링 전략 비교")
    print("="*60)

    n_images = features.shape[0]
    n_samples = 15

    # 1. 랜덤 샘플링
    print("\n1. 랜덤 샘플링:")
    random_diversities = []
    for _ in range(n_trials):
        indices = np.random.choice(n_images, size=n_samples, replace=False)
        sampled = features[indices].numpy()

        dists = [np.linalg.norm(sampled[i] - sampled[j])
                 for i in range(n_samples)
                 for j in range(i+1, n_samples)]
        random_diversities.append(np.mean(dists))

    print(f"  평균 다양성: {np.mean(random_diversities):.4f}")

    # 2. Stratified 샘플링 (첫 번째 특징 기준)
    print("\n2. Stratified 샘플링 (brightness 기준):")
    stratified_diversities = []
    for _ in range(n_trials):
        # brightness로 정렬 후 균등하게 샘플링
        sorted_indices = torch.argsort(features[:, 0])
        step = len(sorted_indices) // n_samples
        indices = [sorted_indices[i * step].item() for i in range(n_samples)]
        sampled = features[indices].numpy()

        dists = [np.linalg.norm(sampled[i] - sampled[j])
                 for i in range(n_samples)
                 for j in range(i+1, n_samples)]
        stratified_diversities.append(np.mean(dists))

    print(f"  평균 다양성: {np.mean(stratified_diversities):.4f}")

    print("\n결론:")
    if np.mean(stratified_diversities) > np.mean(random_diversities):
        print("  → Stratified 샘플링이 더 다양함!")
    else:
        print("  → 랜덤 샘플링도 충분히 다양함")

def main():
    print("="*60)
    print("환경 파라미터 유의미성 분석")
    print("="*60)

    # 이미지 로드 및 특징 추출
    try:
        images, features = load_images_and_features(max_images=None)
        print(f"\n로드된 이미지: {len(images)}개")
        print(f"환경 특징 shape: {features.shape}")
    except Exception as e:
        print(f"Error: {e}")
        print("환경 특징 추출 실패. environment_independent 모듈을 확인하세요.")
        return

    # 분석 실행
    analyze_feature_statistics(features)
    analyze_feature_correlation(features)
    analyze_sampling_diversity(features, n_samples=15)
    compare_w_set_strategies(features, n_trials=5)
    analyze_performance_vs_environment()

    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
