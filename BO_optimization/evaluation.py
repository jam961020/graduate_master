"""
수정된 평가 함수: threshold 조정 및 디버그 추가
"""
import json
import numpy as np
import cv2
from pathlib import Path

# Ground Truth 로드
GT_FILE = Path(__file__).parent.parent / "dataset" / "ground_truth.json"

def load_ground_truth():
    if GT_FILE.exists():
        with open(GT_FILE, 'r') as f:
            return json.load(f)
    return {}

GT_LABELS = load_ground_truth()


def evaluate_lp(detected_coords, image, image_name=None, threshold=50.0, debug=False):
    """
    [수정] LP 기반 평가 - threshold 상향 조정
    
    Args:
        threshold: 픽셀 거리 임계값 (기본 50 픽셀)
                   5픽셀은 너무 엄격 → 50픽셀로 완화
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
    
    # GT 선 위의 픽셀들 생성
    gt_pixels = []
    for line in gt_lines:
        pixels = sample_line_pixels(line, num_samples=100)
        gt_pixels.extend(pixels)
    
    gt_pixels = np.array(gt_pixels)
    
    # 검출된 선들도 픽셀로 변환
    detected_pixels = []
    for line in detected_lines:
        pixels = sample_line_pixels(line, num_samples=100)
        detected_pixels.extend(pixels)
    
    detected_pixels = np.array(detected_pixels)
    
    if len(detected_pixels) == 0:
        if debug:
            print(f"[WARN] No detected pixels for {image_name}")
        return 0.0
    
    # LP 계산
    from scipy.spatial.distance import cdist
    
    distances = cdist(gt_pixels, detected_pixels)
    min_distances = distances.min(axis=1)
    
    tp_count = np.sum(min_distances <= threshold)
    
    # Precision & Recall
    precision = tp_count / len(detected_pixels) if len(detected_pixels) > 0 else 0.0
    recall = tp_count / len(gt_pixels) if len(gt_pixels) > 0 else 0.0
    
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    if debug:
        print(f"  {image_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        print(f"    GT pixels: {len(gt_pixels)}, Detected: {len(detected_pixels)}")
        print(f"    TP: {tp_count}, Threshold: {threshold}")
    
    return f1


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


def evaluate_quality(detected_coords, image, image_name=None, metric="lp", debug=False):
    """
    통합 평가 함수
    
    Args:
        metric: "lp" or "endpoint"
        debug: 디버그 출력 여부
    """
    if metric == "lp":
        # threshold를 5픽셀로 상향 조정
        return evaluate_lp(detected_coords, image, image_name, threshold=5.0, debug=debug)
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