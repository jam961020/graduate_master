"""
미분 가능한 목적함수 - BO 효율 향상
"""
import numpy as np
from scipy.spatial.distance import cdist

def soft_lp_score(detected_coords, gt_coords, image, threshold=50.0, sharpness=0.1):
    """
    Soft LP: 미분 가능한 버전
    
    Args:
        threshold: 거리 임계값 (픽셀)
        sharpness: sigmoid 기울기 (작을수록 부드러움)
    """
    gt_lines = extract_lines_from_coords(gt_coords)
    det_lines = extract_lines_from_coords(detected_coords)
    
    if len(gt_lines) == 0 or len(det_lines) == 0:
        return 0.0
    
    # 선 위의 픽셀들
    gt_pixels = []
    for line in gt_lines:
        pixels = sample_line_pixels(line, num_samples=100)
        gt_pixels.extend(pixels)
    gt_pixels = np.array(gt_pixels)
    
    det_pixels = []
    for line in det_lines:
        pixels = sample_line_pixels(line, num_samples=100)
        det_pixels.extend(pixels)
    det_pixels = np.array(det_pixels)
    
    # 거리 계산
    distances = cdist(gt_pixels, det_pixels)
    min_distances = distances.min(axis=1)
    
    # Soft threshold: sigmoid
    # 1.0 if dist=0, 0.5 if dist=threshold, 0.0 if dist=inf
    weights = 1.0 / (1.0 + np.exp(sharpness * (min_distances - threshold)))
    
    # Soft precision & recall
    soft_tp = np.sum(weights)
    precision = soft_tp / len(det_pixels)
    recall = soft_tp / len(gt_pixels)
    
    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1


def angle_distance_score(detected_coords, gt_coords, image):
    """
    각도 + 거리 기반 점수 (완전히 미분 가능)
    """
    gt_lines = extract_lines_from_coords(gt_coords)
    det_lines = extract_lines_from_coords(detected_coords)
    
    if len(gt_lines) == 0 or len(det_lines) == 0:
        return 0.0
    
    h, w = image.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    
    scores = []
    
    for gt_line in gt_lines:
        x1, y1, x2, y2 = gt_line
        
        # GT 선의 각도
        gt_angle = np.arctan2(y2 - y1, x2 - x1)
        
        # GT 선의 중점
        gt_mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        best_score = 0.0
        
        for det_line in det_lines:
            dx1, dy1, dx2, dy2 = det_line
            
            # 검출 선의 각도
            det_angle = np.arctan2(dy2 - dy1, dx2 - dx1)
            
            # 각도 차이 (0~π)
            angle_diff = abs(gt_angle - det_angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            # 각도 점수 (0~1, 미분 가능)
            angle_score = np.exp(-angle_diff / (np.pi / 4))  # π/4 기준
            
            # 거리 점수
            det_mid = np.array([(dx1 + dx2) / 2, (dy1 + dy2) / 2])
            distance = np.linalg.norm(gt_mid - det_mid)
            distance_score = np.exp(-distance / (diagonal * 0.1))  # 대각선 10% 기준
            
            # 조합
            combined_score = angle_score * distance_score
            
            if combined_score > best_score:
                best_score = combined_score
        
        scores.append(best_score)
    
    return np.mean(scores)


def line_iou_score(detected_coords, gt_coords, image):
    """
    선분 IOU (Intersection over Union)
    """
    gt_lines = extract_lines_from_coords(gt_coords)
    det_lines = extract_lines_from_coords(detected_coords)
    
    if len(gt_lines) == 0 or len(det_lines) == 0:
        return 0.0
    
    ious = []
    
    for gt_line in gt_lines:
        # GT 선 픽셀들
        gt_pixels = set(map(tuple, sample_line_pixels(gt_line, num_samples=200)))
        
        best_iou = 0.0
        
        for det_line in det_lines:
            # 검출 선 픽셀들
            det_pixels = set(map(tuple, sample_line_pixels(det_line, num_samples=200)))
            
            # IOU 계산
            intersection = len(gt_pixels & det_pixels)
            union = len(gt_pixels | det_pixels)
            
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou:
                best_iou = iou
        
        ious.append(best_iou)
    
    return np.mean(ious)


# Helper functions
def extract_lines_from_coords(coords):
    """좌표에서 선분 추출"""
    lines = []
    
    if coords.get("longi_left_upper_x", 0) != 0:
        lines.append([
            coords["longi_left_lower_x"], coords["longi_left_lower_y"],
            coords["longi_left_upper_x"], coords["longi_left_upper_y"]
        ])
    
    if coords.get("longi_right_upper_x", 0) != 0:
        lines.append([
            coords["longi_right_lower_x"], coords["longi_right_lower_y"],
            coords["longi_right_upper_x"], coords["longi_right_upper_y"]
        ])
    
    if coords.get("longi_left_lower_x", 0) != 0 and coords.get("longi_right_lower_x", 0) != 0:
        lines.append([
            coords["longi_left_lower_x"], coords["longi_left_lower_y"],
            coords["longi_right_lower_x"], coords["longi_right_lower_y"]
        ])
    
    if coords.get("collar_left_upper_x", 0) != 0:
        lines.append([
            coords["collar_left_lower_x"], coords["collar_left_lower_y"],
            coords["collar_left_upper_x"], coords["collar_left_upper_y"]
        ])
    
    return lines


def sample_line_pixels(line, num_samples=100):
    """선분 위의 픽셀 샘플링"""
    x1, y1, x2, y2 = line
    t = np.linspace(0, 1, num_samples)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    return np.stack([xs, ys], axis=1)
