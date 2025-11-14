"""
환경 특징 추출 (PSNR/SSIM 포함)

기존 9D baseline + 4D CLIP + 2D Quality Metrics = 15D
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def extract_quality_metrics(roi_image):
    """
    PSNR/SSIM 기반 이미지 품질 메트릭 추출

    Args:
        roi_image: BGR 이미지 (ROI 또는 전체)

    Returns:
        dict: {'psnr': float, 'ssim': float}
    """
    # Grayscale 변환
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image

    # Reference: Gaussian blurred version (노이즈 제거된 이상적 버전)
    # 이렇게 하면 PSNR/SSIM이 낮을수록 원본에 노이즈/왜곡이 많다는 의미
    reference = cv2.GaussianBlur(gray, (5, 5), 0)

    # PSNR 계산 (Peak Signal-to-Noise Ratio)
    # 높을수록 원본과 reference가 유사 (노이즈 적음)
    # 일반적 범위: 20-40 dB (높을수록 좋음)
    psnr = peak_signal_noise_ratio(gray, reference, data_range=255)

    # SSIM 계산 (Structural Similarity Index)
    # 0-1 범위, 높을수록 구조적으로 유사 (왜곡 적음)
    ssim = structural_similarity(gray, reference, data_range=255)

    # 정규화: 낮을수록 어려운 환경 (노이즈 많음)
    # PSNR: 20-50 범위를 0-1로 매핑 (낮을수록 어려움)
    psnr_normalized = np.clip((50 - psnr) / 30, 0.0, 1.0)  # 50이상=0, 20이하=1

    # SSIM: 이미 0-1, 반전 (낮을수록 어려움)
    ssim_normalized = 1.0 - ssim

    return {
        'psnr_score': float(psnr_normalized),  # 높을수록 노이즈 많음 (어려움)
        'ssim_score': float(ssim_normalized)   # 높을수록 구조 왜곡 (어려움)
    }


def test_quality_metrics():
    """테스트 함수"""
    import matplotlib.pyplot as plt

    # 테스트 이미지 생성
    print("Testing PSNR/SSIM metrics...")

    # 1. Clean image
    clean = np.ones((100, 100), dtype=np.uint8) * 128

    # 2. Noisy image
    noisy = clean.copy()
    noise = np.random.normal(0, 30, (100, 100))
    noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)

    # 3. Blurred image
    blurred = cv2.GaussianBlur(clean, (21, 21), 0)

    print("\n1. Clean image:")
    metrics = extract_quality_metrics(clean)
    print(f"   PSNR score: {metrics['psnr_score']:.4f} (lower=better quality)")
    print(f"   SSIM score: {metrics['ssim_score']:.4f} (lower=better quality)")

    print("\n2. Noisy image:")
    metrics = extract_quality_metrics(noisy)
    print(f"   PSNR score: {metrics['psnr_score']:.4f}")
    print(f"   SSIM score: {metrics['ssim_score']:.4f}")

    print("\n3. Blurred image:")
    metrics = extract_quality_metrics(blurred)
    print(f"   PSNR score: {metrics['psnr_score']:.4f}")
    print(f"   SSIM score: {metrics['ssim_score']:.4f}")

    print("\nExpected: Noisy > Clean, Blurred > Clean")
    print("✓ Test complete!")


if __name__ == "__main__":
    test_quality_metrics()
