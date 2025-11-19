"""
수정된 평가 함수: threshold 조정 및 디버그 추가
"""
import json
import numpy as np
import cv2
from pathlib import Path

# Ground Truth 로드
GT_FILE = Path(__file__).parent.parent / "dataset" / "ground_truth_merged.json"

def load_ground_truth():
    if GT_FILE.exists():
        with open(GT_FILE, 'r') as f:
            return json.load(f)
    return {}

GT_LABELS = load_ground_truth()


def evaluate_lp(detected_coords, image, image_name=None, threshold=30.0, debug=False):
    """
    AirLine 논문의 LP_r (Line Precision) 구현

    Reference: "AirLine: Efficient Learnable Line Detection with Local Edge Voting" (IROS 2023)

    LP_r = Σ(τ_r(X) ⊗ Y) / ΣY

    where:
        - X: detected lines (검출된 선)
        - Y: ground truth pixels (GT 선의 픽셀들)
        - τ_r: dilation function with tolerance radius r
        - ⊗: element-wise multiplication (overlap)

    의미: GT 픽셀 중 검출된 선으로부터 r 픽셀 이내에 있는 비율 (Recall)

    Args:
        detected_coords: 검출된 좌표 딕셔너리
        image: 이미지 (사용 안 함, 호환성 유지)
        image_name: 이미지 이름
        threshold: tolerance radius r (픽셀 단위, 기본 50)
        debug: 디버그 출력 여부

    Returns:
        LP_r score (0~1): GT coverage ratio

    Note:
        - "Line Precision"이라는 이름이지만 실제로는 Recall 측정
        - RANSAC 후 단일 선만 남으므로 over-detection 문제 최소화
        - Precision/F1은 사용하지 않음 (논문 원본 그대로)
    """
    if image_name is None:
        return 0.0

    # 증강 이미지 이름 처리
    base_name = image_name.split('_aug')[0]

    if base_name not in GT_LABELS:
        if debug:
            print(f"[WARN] No GT for {image_name} (base: {base_name})")
        return 0.0

    gt_coords = GT_LABELS[base_name].get("coordinates", GT_LABELS[base_name])

    # GT와 검출 결과에서 실제로 존재하는 선들만 추출
    gt_lines = extract_lines_from_coords(gt_coords)
    detected_lines = extract_lines_from_coords(detected_coords)

    if len(gt_lines) == 0:
        if debug:
            print(f"[WARN] No GT lines for {image_name}")
        return 0.0

    if len(detected_lines) == 0:
        if debug:
            print(f"[WARN] No detected lines for {image_name}")
        return 0.0

    # GT 선들을 픽셀로 샘플링
    gt_pixels = []
    for line in gt_lines:
        pixels = sample_line_pixels(line, num_samples=100)
        gt_pixels.extend(pixels)

    gt_pixels = np.array(gt_pixels)

    # 검출된 선들을 픽셀로 샘플링
    detected_pixels = []
    for line in detected_lines:
        pixels = sample_line_pixels(line, num_samples=100)
        detected_pixels.extend(pixels)

    detected_pixels = np.array(detected_pixels)

    if len(detected_pixels) == 0:
        if debug:
            print(f"[WARN] No detected pixels for {image_name}")
        return 0.0

    # LP_r 계산: GT 픽셀 → 검출된 픽셀까지의 최소 거리
    from scipy.spatial.distance import cdist

    # distances[i, j] = GT 픽셀 i와 검출 픽셀 j 사이의 거리
    distances = cdist(gt_pixels, detected_pixels)

    # 각 GT 픽셀에서 가장 가까운 검출 픽셀까지의 거리
    min_distances = distances.min(axis=1)

    # 연속 점수: 거리 0 → 1.0, threshold → 0.0, 초과 → 0.0 (선형 감쇠)
    # 이전: 이진 평가 (threshold 이내만 1, 초과는 0)
    # 현재: 거리 비례 점수 (더 부드러운 gradient, BO 최적화에 유리)
    pixel_scores = np.clip(1.0 - min_distances / threshold, 0.0, 1.0)

    # LP_r = 모든 GT 픽셀의 평균 점수
    lp_r = pixel_scores.mean()

    if debug:
        # 통계 정보: threshold 이내 픽셀 수도 같이 표시
        covered_count = np.sum(min_distances <= threshold)
        print(f"  {image_name}: LP_{int(threshold)}={lp_r:.4f} (avg score, {covered_count}/{len(gt_pixels)} within threshold)")
        print(f"    GT lines: {len(gt_lines)}, Detected lines: {len(detected_lines)}")
        print(f"    GT pixels: {len(gt_pixels)}, Detected pixels: {len(detected_pixels)}")
        print(f"    Distance stats: min={min_distances.min():.1f}, mean={min_distances.mean():.1f}, max={min_distances.max():.1f}")

    return lp_r


def evaluate_endpoint_error(detected_coords, image, image_name=None, debug=False):
    """
    [수정] Endpoint Error - 검출된 좌표만 평가
    """
    if image_name is None:
        return 0.0
    
    # 증강 이미지 이름 처리
    base_name = image_name.split('_aug')[0]
    
    if base_name not in GT_LABELS:
        if debug:
            print(f"[WARN] No GT for {image_name} (base: {base_name})")
        return 0.0
    
    gt_coords = GT_LABELS[base_name].get("coordinates", GT_LABELS[base_name])
    
    gt_lines = extract_lines_from_coords(gt_coords)
    detected_lines = extract_lines_from_coords(detected_coords)
    
    if len(gt_lines) == 0 or len(detected_lines) == 0:
        return 0.0
    
    h, w = image.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    
    errors = []
    
    for gt_line in gt_lines:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_line
        
        min_error = float('inf')
        
        for det_line in detected_lines:
            det_x1, det_y1, det_x2, det_y2 = det_line
            
            # 끝점 거리 계산
            dist1_start = np.sqrt((gt_x1 - det_x1)**2 + (gt_y1 - det_y1)**2)
            dist1_end = np.sqrt((gt_x2 - det_x2)**2 + (gt_y2 - det_y2)**2)
            error1 = (dist1_start + dist1_end) / 2
            
            dist2_start = np.sqrt((gt_x1 - det_x2)**2 + (gt_y1 - det_y2)**2)
            dist2_end = np.sqrt((gt_x2 - det_x1)**2 + (gt_y2 - det_y1)**2)
            error2 = (dist2_start + dist2_end) / 2
            
            error = min(error1, error2)
            
            if error < min_error:
                min_error = error
        
        errors.append(min_error)
    
    mean_error = np.mean(errors)
    normalized_error = mean_error / diagonal
    score = np.exp(-normalized_error * 10)
    
    if debug:
        print(f"  {image_name}: Mean error={mean_error:.1f}px, Score={score:.3f}")
    
    return np.clip(score, 0.0, 1.0)


def extract_lines_from_coords(coords):
    """0이 아닌 좌표만 추출"""
    lines = []
    
    # 1. Left Longi
    if coords.get("longi_left_upper_x", 0) != 0 and coords.get("longi_left_lower_x", 0) != 0:
        lines.append([
            coords["longi_left_lower_x"],
            coords["longi_left_lower_y"],
            coords["longi_left_upper_x"],
            coords["longi_left_upper_y"]
        ])
    
    # 2. Right Longi
    if coords.get("longi_right_upper_x", 0) != 0 and coords.get("longi_right_lower_x", 0) != 0:
        lines.append([
            coords["longi_right_lower_x"],
            coords["longi_right_lower_y"],
            coords["longi_right_upper_x"],
            coords["longi_right_upper_y"]
        ])
    
    # 3. Fillet (가로선)
    if coords.get("longi_left_lower_x", 0) != 0 and coords.get("longi_right_lower_x", 0) != 0:
        lines.append([
            coords["longi_left_lower_x"],
            coords["longi_left_lower_y"],
            coords["longi_right_lower_x"],
            coords["longi_right_lower_y"]
        ])
    
    # 4. Collar
    if coords.get("collar_left_upper_x", 0) != 0 and coords.get("collar_left_lower_x", 0) != 0:
        lines.append([
            coords["collar_left_lower_x"],
            coords["collar_left_lower_y"],
            coords["collar_left_upper_x"],
            coords["collar_left_upper_y"]
        ])
    
    return lines


def sample_line_pixels(line, num_samples=100):
    """선분 위의 픽셀들을 균등하게 샘플링"""
    x1, y1, x2, y2 = line
    
    t = np.linspace(0, 1, num_samples)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    
    pixels = np.stack([xs, ys], axis=1)
    
    return pixels


def evaluate_quality(detected_coords, image, image_name=None, metric="lp", threshold=20.0, debug=False):
    """
    통합 평가 함수

    Args:
        metric: "lp" or "endpoint"
        threshold: tolerance radius for lp metric (default: 20.0 pixels)
        debug: 디버그 출력 여부
    """
    if metric == "lp":
        # AirLine 논문 원본 LP_r 사용
        # threshold=20: 적당한 tolerance (이미지 해상도 2448×3264 고려)
        return evaluate_lp(detected_coords, image, image_name, threshold=threshold, debug=debug)
    elif metric == "endpoint":
        return evaluate_endpoint_error(detected_coords, image, image_name, debug=debug)
    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    # 테스트 코드
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <image_name>")
        sys.exit(1)
    
    image_name = sys.argv[1]
    
    if image_name not in GT_LABELS:
        print(f"No GT for {image_name}")
        sys.exit(1)
    
    gt_coords = GT_LABELS[image_name].get("coordinates", GT_LABELS[image_name])
    
    # 테스트: GT를 detected로 사용 (완벽한 검출)
    print(f"Testing with GT as detected coords...")
    
    # 더미 이미지
    test_image = np.zeros((2448, 3264, 3), dtype=np.uint8)
    
    lp_score = evaluate_lp(gt_coords, test_image, image_name, threshold=50.0, debug=True)
    endpoint_score = evaluate_endpoint_error(gt_coords, test_image, image_name, debug=True)
    
    print(f"\nLP Score (threshold=50):  {lp_score:.4f}")
    print(f"Endpoint Score:            {endpoint_score:.4f}")