"""
환경변수 추출 (파라미터 독립적)

핵심 원칙:
- 결정변수(edgeThresh, simThresh 등)와 완전히 독립적
- 이미지의 본질적 특성만 사용
- 고정된 알고리즘 파라미터만 사용
"""
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def extract_parameter_independent_environment(image, roi=None):
    """
    파라미터와 독립적인 이미지 환경 특성 추출
    
    Args:
        image: BGR 이미지
        roi: (x1, y1, x2, y2) 또는 None (전체)
    
    Returns:
        env: {
            'brightness': float [0, 1],
            'contrast': float [0, 1],
            'edge_density': float [0, 1],
            'texture_complexity': float [0, 1],
            'blur_level': float [0, 1],
            'noise_level': float [0, 1]
        }
    """
    # ROI 추출
    if roi:
        x1, y1, x2, y2 = roi
        img = image[y1:y2, x1:x2]
    else:
        img = image
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 밝기 (Brightness)
    # 중간값(128)에서 멀수록 어려움
    brightness_mean = gray.mean()
    brightness_score = abs(brightness_mean - 128) / 128  # 0=ideal, 1=extreme
    
    # 2. 대비 (Contrast)
    # 낮을수록 어려움
    contrast_range = gray.max() - gray.min()
    contrast_score = 1.0 - (contrast_range / 255.0)  # 0=high contrast, 1=low contrast
    
    # 3. 엣지 밀도 (Edge Density)
    # 고정 파라미터 Canny 사용 (파라미터 독립적!)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (edges.size * 255)
    # 적당한 밀도(0.1~0.3)가 이상적, 너무 많거나 적으면 어려움
    if edge_density < 0.1:
        edge_score = 1.0 - (edge_density / 0.1)  # 너무 적음
    elif edge_density > 0.3:
        edge_score = (edge_density - 0.3) / 0.7  # 너무 많음 (노이즈)
    else:
        edge_score = 0.0  # 이상적
    edge_score = np.clip(edge_score, 0.0, 1.0)
    
    # 4. 텍스처 복잡도 (Texture Complexity)
    # Laplacian variance로 측정
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_var = laplacian.var()
    # 높을수록 복잡 → 어려움
    texture_score = np.clip(texture_var / 1000, 0.0, 1.0)
    
    # 5. 블러 레벨 (Blur Level)
    # Laplacian variance 낮으면 블러됨
    blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = 1.0 - np.clip(blur_variance / 500, 0.0, 1.0)
    
    # 6. 노이즈 레벨 (Noise Level)
    # 고주파 성분 비율로 추정
    # Gaussian blur 후 차이
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(float) - blurred.astype(float))
    noise_score = np.clip(noise.mean() / 50, 0.0, 1.0)
    
    env = {
        'brightness': float(brightness_score),
        'contrast': float(contrast_score),
        'edge_density': float(edge_score),
        'texture_complexity': float(texture_score),
        'blur_level': float(blur_score),
        'noise_level': float(noise_score)
    }
    
    return env


def compute_difficulty_score(env, weights=None):
    """
    환경 특성으로부터 종합 난이도 계산
    
    Args:
        env: 환경 dict
        weights: 가중치 dict (None이면 균등)
    
    Returns:
        difficulty: float [0, 1]
    """
    if weights is None:
        weights = {
            'brightness': 0.15,
            'contrast': 0.20,
            'edge_density': 0.20,
            'texture_complexity': 0.15,
            'blur_level': 0.15,
            'noise_level': 0.15
        }
    
    difficulty = sum(env[k] * weights[k] for k in weights.keys())
    return float(difficulty)


def batch_extract_environments(image_dir, output_file, yolo_detector=None):
    """
    데이터셋 전체에 대해 환경변수 추출
    
    Args:
        image_dir: 이미지 디렉토리
        output_file: 출력 JSON 파일
        yolo_detector: YOLO 검출기 (ROI 사용 시)
    """
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    results = {}
    
    for img_path in tqdm(images, desc="Extracting environments"):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # 전체 이미지에 대해 환경 추출
            env = extract_parameter_independent_environment(image, roi=None)
            
            # 난이도 계산
            difficulty = compute_difficulty_score(env)
            env['difficulty'] = difficulty
            
            results[img_path.stem] = env
            
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue
    
    # 저장
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    
    # 통계
    if results:
        difficulties = [data['difficulty'] for data in results.values()]
        brightnesses = [data['brightness'] for data in results.values()]
        contrasts = [data['contrast'] for data in results.values()]
        
        print(f"\n=== 환경 통계 ({len(results)}장) ===")
        print(f"종합 난이도: {min(difficulties):.3f} ~ {max(difficulties):.3f} (평균: {np.mean(difficulties):.3f})")
        print(f"밝기 문제:   {min(brightnesses):.3f} ~ {max(brightnesses):.3f} (평균: {np.mean(brightnesses):.3f})")
        print(f"대비 문제:   {min(contrasts):.3f} ~ {max(contrasts):.3f} (평균: {np.mean(contrasts):.3f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="이미지 디렉토리")
    parser.add_argument("--output", default="environment_independent.json", help="출력 파일")
    args = parser.parse_args()
    
    batch_extract_environments(args.input_dir, args.output)
