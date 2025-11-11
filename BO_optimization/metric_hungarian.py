"""
직선 유사도 메트릭 (헝가리안 알고리즘 기반)

특징:
- 각도 유사도 + 거리 패널티
- 여러 선분 간 최적 매칭 (헝가리안 알고리즘)
- ROI 기반 선분 그룹화 지원
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


def compute_line_angle(line):
    """
    선분의 각도 계산
    
    Args:
        line: [x1, y1, x2, y2]
    
    Returns:
        angle: radian [-π, π]
    """
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1)
    return angle


def compute_line_midpoint(line):
    """
    선분의 중점 계산
    
    Args:
        line: [x1, y1, x2, y2]
    
    Returns:
        midpoint: [x, y]
    """
    x1, y1, x2, y2 = line
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def line_similarity(line1, line2, angle_weight=0.6, distance_weight=0.4, max_distance=100):
    """
    두 선분 간 유사도 계산
    
    Args:
        line1: [x1, y1, x2, y2] 검출 선분
        line2: [x1, y1, x2, y2] GT 선분
        angle_weight: 각도 유사도 가중치
        distance_weight: 거리 유사도 가중치
        max_distance: 최대 거리 (정규화용)
    
    Returns:
        similarity: float [0, 1] (1이 완벽)
    """
    # 1. 각도 유사도
    angle1 = compute_line_angle(line1)
    angle2 = compute_line_angle(line2)
    
    angle_diff = abs(angle1 - angle2)
    # 각도는 주기적이므로 180도 이상이면 반대편으로
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    
    angle_similarity = 1.0 - (angle_diff / np.pi)
    
    # 2. 거리 유사도 (중점 간 거리)
    mid1 = compute_line_midpoint(line1)
    mid2 = compute_line_midpoint(line2)
    
    distance = np.linalg.norm(mid1 - mid2)
    distance_similarity = max(0, 1.0 - (distance / max_distance))
    
    # 3. 종합 유사도
    similarity = angle_weight * angle_similarity + distance_weight * distance_similarity
    
    return similarity


def hungarian_line_matching(detected_lines, gt_lines, angle_weight=0.6, distance_weight=0.4):
    """
    헝가리안 알고리즘으로 검출 선분과 GT 선분 최적 매칭
    
    Args:
        detected_lines: List of [x1, y1, x2, y2] (N개)
        gt_lines: List of [x1, y1, x2, y2] (M개)
        angle_weight: 각도 가중치
        distance_weight: 거리 가중치
    
    Returns:
        matched_pairs: List of (det_idx, gt_idx, similarity)
        unmatched_det: List of det_idx (매칭 안된 검출)
        unmatched_gt: List of gt_idx (매칭 안된 GT)
        avg_similarity: 매칭된 쌍들의 평균 유사도
    """
    if len(detected_lines) == 0 or len(gt_lines) == 0:
        return [], list(range(len(detected_lines))), list(range(len(gt_lines))), 0.0
    
    # 코스트 매트릭스 생성 (N x M)
    # 코스트 = 1 - similarity (최소화 문제로 변환)
    cost_matrix = np.zeros((len(detected_lines), len(gt_lines)))
    
    for i, det_line in enumerate(detected_lines):
        for j, gt_line in enumerate(gt_lines):
            similarity = line_similarity(det_line, gt_line, angle_weight, distance_weight)
            cost_matrix[i, j] = 1.0 - similarity  # 코스트로 변환
    
    # 헝가리안 알고리즘
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 매칭 결과 구성
    matched_pairs = []
    for det_idx, gt_idx in zip(row_ind, col_ind):
        similarity = 1.0 - cost_matrix[det_idx, gt_idx]
        # 유사도가 너무 낮으면 매칭 취소 (threshold)
        if similarity > 0.3:  # 임계값
            matched_pairs.append((det_idx, gt_idx, similarity))
    
    # 매칭 안된 인덱스
    matched_det_indices = {pair[0] for pair in matched_pairs}
    matched_gt_indices = {pair[1] for pair in matched_pairs}
    
    unmatched_det = [i for i in range(len(detected_lines)) if i not in matched_det_indices]
    unmatched_gt = [i for i in range(len(gt_lines)) if i not in matched_gt_indices]
    
    # 평균 유사도
    if matched_pairs:
        avg_similarity = np.mean([sim for _, _, sim in matched_pairs])
    else:
        avg_similarity = 0.0
    
    return matched_pairs, unmatched_det, unmatched_gt, avg_similarity


def evaluate_detection_hungarian(detected_coords, gt_coords, image=None, image_name=""):
    """
    헝가리안 알고리즘 기반 검출 평가
    
    Args:
        detected_coords: dict {
            'longi_left_upper_x': float, ...
            'collar_left_lower_x': float, ...
        }
        gt_coords: dict (동일 형식)
        image: BGR 이미지 (옵션)
        image_name: 이미지 이름 (디버깅용)
    
    Returns:
        score: float [0, 1] (1이 완벽)
    """
    # 좌표를 선분으로 변환
    def coords_to_lines(coords):
        lines = []
        
        # longi left
        if all(coords.get(k, 0) != 0 for k in 
               ['longi_left_lower_x', 'longi_left_lower_y', 
                'longi_left_upper_x', 'longi_left_upper_y']):
            lines.append([
                coords['longi_left_lower_x'], coords['longi_left_lower_y'],
                coords['longi_left_upper_x'], coords['longi_left_upper_y']
            ])
        
        # longi right
        if all(coords.get(k, 0) != 0 for k in 
               ['longi_right_lower_x', 'longi_right_lower_y', 
                'longi_right_upper_x', 'longi_right_upper_y']):
            lines.append([
                coords['longi_right_lower_x'], coords['longi_right_lower_y'],
                coords['longi_right_upper_x'], coords['longi_right_upper_y']
            ])
        
        # collar
        if all(coords.get(k, 0) != 0 for k in 
               ['collar_left_lower_x', 'collar_left_lower_y', 
                'collar_left_upper_x', 'collar_left_upper_y']):
            lines.append([
                coords['collar_left_lower_x'], coords['collar_left_lower_y'],
                coords['collar_left_upper_x'], coords['collar_left_upper_y']
            ])
        
        return lines
    
    detected_lines = coords_to_lines(detected_coords)
    gt_lines = coords_to_lines(gt_coords)
    
    if len(gt_lines) == 0:
        return 0.0  # GT 없으면 평가 불가
    
    if len(detected_lines) == 0:
        return 0.0  # 검출 실패
    
    # 헝가리안 매칭
    matched_pairs, unmatched_det, unmatched_gt, avg_similarity = hungarian_line_matching(
        detected_lines, gt_lines
    )
    
    # 점수 계산
    # 1. 매칭된 선분의 평균 유사도
    # 2. 매칭 안된 GT에 대한 패널티
    # 3. False positive (매칭 안된 검출)에 대한 경미한 패널티
    
    n_matched = len(matched_pairs)
    n_gt = len(gt_lines)
    
    if n_matched == 0:
        return 0.0
    
    # 기본 점수: 매칭된 쌍의 평균 유사도
    base_score = avg_similarity
    
    # 재현율 패널티: 매칭 안된 GT 비율
    recall_penalty = len(unmatched_gt) / n_gt
    
    # 정밀도 패널티: 매칭 안된 검출 비율 (더 가벼운 패널티)
    precision_penalty = 0.3 * (len(unmatched_det) / len(detected_lines))
    
    # 최종 점수
    score = base_score * (1.0 - recall_penalty) * (1.0 - precision_penalty)
    
    return float(np.clip(score, 0.0, 1.0))


def evaluate_with_multiple_lines(detected_lines_dict, gt_lines_dict):
    """
    AirLine 결과로 나온 여러 선분과 GT 매칭
    
    Args:
        detected_lines_dict: {
            'longi': [[x1,y1,x2,y2], ...],
            'collar': [[x1,y1,x2,y2], ...]
        }
        gt_lines_dict: 동일 형식
    
    Returns:
        score: float [0, 1]
    """
    total_score = 0.0
    total_weight = 0.0
    
    for key in ['longi', 'collar']:
        if key not in detected_lines_dict or key not in gt_lines_dict:
            continue
        
        det_lines = detected_lines_dict[key]
        gt_lines = gt_lines_dict[key]
        
        if len(gt_lines) == 0:
            continue
        
        _, _, _, avg_sim = hungarian_line_matching(det_lines, gt_lines)
        
        # 가중치: GT 선분 수에 비례
        weight = len(gt_lines)
        total_score += avg_sim * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_score / total_weight


if __name__ == "__main__":
    # 테스트
    print("=== 직선 유사도 메트릭 테스트 ===\n")
    
    # 예시 1: 완벽한 매칭
    det_lines = [
        [100, 100, 200, 200],  # 45도
        [100, 200, 200, 100],  # -45도
    ]
    gt_lines = [
        [105, 105, 205, 205],  # 45도, 약간 이동
        [95, 195, 195, 95],    # -45도, 약간 이동
    ]
    
    matched, unmatched_det, unmatched_gt, avg_sim = hungarian_line_matching(det_lines, gt_lines)
    print(f"테스트 1: 완벽한 매칭")
    print(f"  매칭된 쌍: {len(matched)}")
    print(f"  평균 유사도: {avg_sim:.3f}")
    print(f"  매칭 정보: {matched}\n")
    
    # 예시 2: 부분 매칭
    det_lines = [
        [100, 100, 200, 200],
        [300, 300, 400, 400],  # 매칭 안될 것
    ]
    gt_lines = [
        [105, 105, 205, 205],
        [500, 500, 600, 600],  # 매칭 안될 것
        [700, 700, 800, 800],  # 매칭 안될 것
    ]
    
    matched, unmatched_det, unmatched_gt, avg_sim = hungarian_line_matching(det_lines, gt_lines)
    print(f"테스트 2: 부분 매칭")
    print(f"  매칭된 쌍: {len(matched)}")
    print(f"  매칭 안된 검출: {unmatched_det}")
    print(f"  매칭 안된 GT: {unmatched_gt}")
    print(f"  평균 유사도: {avg_sim:.3f}")
