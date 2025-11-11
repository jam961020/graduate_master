"""
전체 파이프라인: YOLO + AirLine + 교점 계산
test_clone.py의 로직을 BO용으로 래핑

수정사항:
- process_guideline_roi에 params 인자 추가
- process_collar_roi에 params 인자 추가 (일관성)
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# 경로 추가
YOLO_PATH = Path(__file__).parent.parent / "YOLO_AirLine"
sys.path.insert(0, str(YOLO_PATH))

# AirLine_assemble_test에서 import
try:
    from AirLine_assemble_test import (
        run_airline, run_lsd, run_fld, run_hough,
        enhance_color, get_line_pixels, extend_line,
        filter_lines_by_diagonal, filter_line_by_centrality,
        find_best_fit_line_ransac, get_intersection,
        sharp_S
    )
    print("[INFO] Imported from AirLine_assemble_test.py successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import from AirLine_assemble_test: {e}")
    raise

from yolo_detector import YOLODetector


def find_upper_point_by_intersection(longi_res):
    """AirLine 수직 교차선을 이용해 Upper Point를 찾습니다."""
    if not longi_res:
        return None
    
    longi_line = np.array(longi_res["line"])
    roi = longi_res["roi"]
    x1_r, y1_r, x2_r, y2_r = roi
    airline_lines_roi = longi_res["airline_lines_in_roi"]
    
    if airline_lines_roi.size == 0:
        return None
    
    longi_dx = longi_line[2] - longi_line[0]
    longi_dy = longi_line[3] - longi_line[1]
    longi_angle = np.rad2deg(np.mod(np.arctan2(longi_dy, longi_dx), np.pi))
    
    candidate_intersections = []
    roi_height = y2_r - y1_r
    upper_region_y_threshold = roi_height * 0.15
    
    for al_line in airline_lines_roi:
        al_dx = al_line[2] - al_line[0]
        al_dy = al_line[3] - al_line[1]
        al_angle = np.rad2deg(np.mod(np.arctan2(al_dy, al_dx), np.pi))
        
        angle_diff = abs(longi_angle - al_angle)
        if abs(angle_diff - 90) < 15:
            midpoint_y = (al_line[1] + al_line[3]) / 2
            if midpoint_y < upper_region_y_threshold:
                al_line_global = al_line + [x1_r, y1_r, x1_r, y1_r]
                pt = get_intersection(longi_line, al_line_global)
                if pt:
                    candidate_intersections.append(pt)
    
    if candidate_intersections:
        return min(candidate_intersections, key=lambda p: p[1])
    return None


def find_upper_point_by_roi_fallback(longi_res):
    """ROI 경계를 기준으로 Upper Point 폴백 로직을 수행합니다."""
    if not longi_res:
        return None
    
    longi_line = longi_res["line"]
    x1_r, y1_r, x2_r, y2_r = longi_res["roi"]
    lx1, ly1, lx2, ly2 = longi_line
    
    offset = (y2_r - y1_r) * 0.035
    
    if (lx2 - lx1) == 0:
        return (int(lx1), int(y1_r + offset))
    else:
        m = (ly2 - ly1) / (lx2 - lx1)
        c = ly1 - m * lx1
        y_intersect = y1_r
        x_intersect = (y_intersect - c) / m
        return (int(x_intersect), int(y_intersect + offset))


def find_3_sides_of_collar(lines_by_algo, roi_w, roi_h, is_left_side):
    """
    Collar의 3변 (top, bottom, outer_vertical) 찾기
    
    Args:
        lines_by_algo: 알고리즘별 검출된 선들
        roi_w, roi_h: ROI 크기
        is_left_side: 왼쪽 칼라 여부
    
    Returns:
        dict with keys: 'top', 'bottom', 'outer_vertical'
    """
    merged = []
    for key in ("AirLine_Q", "AirLine_QG"):
        arr = lines_by_algo.get(key)
        if arr is not None and arr.size > 0:
            merged.append(arr)
    
    if merged:
        airline_lines = np.vstack(merged)
    else:
        merged_fb = []
        for key in ("FLD", "LSD", "Hough"):
            arr = lines_by_algo.get(key)
            if arr is not None and arr.size > 0:
                merged_fb.append(arr)
        if not merged_fb:
            return {}
        airline_lines = np.vstack(merged_fb)
    
    airline_lines = np.clip(airline_lines, [0, 0, 0, 0], [roi_w, roi_h, roi_w, roi_h])
    
    directional_groups = filter_lines_by_diagonal(airline_lines, roi_w, roi_h)
    h_lines = directional_groups['horizontal']
    v_lines = directional_groups['vertical']
    
    top_lines = [l for l in h_lines if max(l[1], l[3]) < roi_h * 0.2]
    bottom_lines = [l for l in h_lines if min(l[1], l[3]) > roi_h * 0.8]
    
    top_line = max(top_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), -max(l[1], l[3])), default=None)
    bottom_line = max(bottom_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), min(l[1], l[3])), default=None)
    
    outer_v_line = None
    if len(v_lines) > 0:
        if is_left_side:
            right_lines = [l for l in v_lines if min(l[0], l[2]) > roi_w * 0.7]
            outer_v_line = max(right_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), (l[0]+l[2])/2), default=None)
        else:
            left_lines = [l for l in v_lines if max(l[0], l[2]) < roi_w * 0.3]
            outer_v_line = max(left_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), -(l[0]+l[2])/2), default=None)
    
    found_edges = {}
    if top_line is not None:
        found_edges['top'] = top_line
    if bottom_line is not None:
        found_edges['bottom'] = bottom_line
    if outer_v_line is not None:
        found_edges['outer_vertical'] = outer_v_line
    
    return found_edges


def weighted_ransac_line(final_candidates, roi_w, roi_h,
                         w_center=0.5, w_length=0.5, consensus_weight=5):
    """
    Weighted RANSAC for line selection from multiple algorithm candidates.
    """
    all_lines = []
    for algo, lines in (final_candidates or {}).items():
        # ✅ numpy array와 list 모두 처리
        if lines is None or len(lines) == 0:
            continue
        for ln in lines:
            all_lines.append((str(algo), ln))

    if not all_lines:
        return None

    def line_len(ln):
        x1,y1,x2,y2 = ln
        return float(np.hypot(x2-x1, y2-y1))

    def center_weight(ln):
        x1,y1,x2,y2 = ln
        cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
        dx = abs(cx - roi_w/2)/(roi_w/2+1e-6)
        dy = abs(cy - roi_h/2)/(roi_h/2+1e-6)
        return float(np.clip(1.0 - 0.5*(dx+dy), 0.0, 1.0))

    lengths = np.array([line_len(ln) for _, ln in all_lines], float)
    cweights = np.array([center_weight(ln) for _, ln in all_lines], float)
    if lengths.max() > 0:
        lengths = lengths / (lengths.max() + 1e-6)

    probs = 1e-6 + (w_length * lengths + w_center * cweights)
    probs = probs / probs.sum()

    algo_tags = np.array([t for t,_ in all_lines])
    airline_mask = (algo_tags == "AirLine_Q") | (algo_tags == "AirLine_QG")
    if airline_mask.any() and consensus_weight > 1:
        probs[airline_mask] *= float(consensus_weight)
        probs = probs / probs.sum()

    pts = []
    for _, ln in all_lines:
        x1,y1,x2,y2 = ln
        pts.append(((x1+x2)/2.0, (y1+y2)/2.0))
    pts = np.array(pts)

    best = None; best_inl = -1
    iters = min(500, max(100, len(all_lines)*4))
    tol = max(2.0, 0.01*max(roi_w, roi_h))

    rng = np.random.default_rng()
    for _ in range(iters):
        i1, i2 = rng.choice(len(all_lines), size=2, replace=False, p=probs)
        (_, l1), (_, l2) = all_lines[i1], all_lines[i2]
        x1,y1,x2,y2 = l1
        X1 = np.array([x1,y1]); X2 = np.array([x2,y2])
        A = X2[1]-X1[1]; B = X1[0]-X2[0]; C = X2[0]*X1[1]-X1[0]*X2[1]
        denom = np.hypot(A,B) + 1e-6
        d = np.abs(A*pts[:,0] + B*pts[:,1] + C)/denom
        inl = int((d < tol).sum())
        if inl > best_inl:
            best_inl = inl
            cand = []
            for ty in (0, roi_h):
                if X1[1] != X2[1]:
                    x = X1[0] + (X2[0]-X1[0])*(ty - X1[1])/(X2[1]-X1[1])
                    if 0 <= x <= roi_w: cand.append((int(x), int(ty)))
            for tx in (0, roi_w):
                if X1[0] != X2[0]:
                    y = X1[1] + (X2[1]-X1[1])*(tx - X1[0])/(X2[0]-X1[0])
                    if 0 <= y <= roi_h: cand.append((int(tx), int(y)))
            if len(cand) >= 2:
                cand = sorted(list(set(cand)))
                best = (cand[0], cand[-1])

    return best


def detect_lines_in_roi(roi_bgr_enhanced, roi_gray_enhanced, params):
    """ROI 내에서 모든 선 검출 알고리즘 실행"""
    lines_by_algo = {}
    
    lines_by_algo["LSD"] = run_lsd(roi_gray_enhanced)
    lines_by_algo["FLD"] = run_fld(roi_gray_enhanced)
    lines_by_algo["Hough"] = run_hough(roi_gray_enhanced)
    
    diagonal = np.sqrt(roi_bgr_enhanced.shape[0]**2 + roi_bgr_enhanced.shape[1]**2)
    
    airline_preset_Q = {
        "edgeThresh": params.get('edgeThresh1', -3.0),
        "simThresh": params.get('simThresh1', 0.98),
        "pixelNumThresh": int(diagonal * params.get('pixelRatio1', 0.05))
    }
    airline_preset_QG = {
        "edgeThresh": params.get('edgeThresh2', 1.0),
        "simThresh": params.get('simThresh2', 0.75),
        "pixelNumThresh": int(diagonal * params.get('pixelRatio2', 0.05))
    }

    print(f"[AirLine] Q   edge={airline_preset_Q['edgeThresh']:.3f}, "
          f"sim={airline_preset_Q['simThresh']:.3f}, "
          f"px={airline_preset_Q['pixelNumThresh']}")
    print(f"[AirLine] QG  edge={airline_preset_QG['edgeThresh']:.3f}, "
          f"sim={airline_preset_QG['simThresh']:.3f}, "
          f"px={airline_preset_QG['pixelNumThresh']}")

    try:
        lines_by_algo["AirLine_Q"] = run_airline(roi_bgr_enhanced, airline_preset_Q)
        lines_by_algo["AirLine_QG"] = run_airline(roi_bgr_enhanced, airline_preset_QG)
    except Exception as e:
        print(f"  [WARN] AirLine failed: {e}")
        lines_by_algo["AirLine_Q"] = np.empty((0, 4))
        lines_by_algo["AirLine_QG"] = np.empty((0, 4))
    
    return lines_by_algo


def process_guideline_roi(lines_by_algo, roi_bgr, cls, roi_bbox, image_width, params):
    """
    클래스 1 (fillet) 또는 2 (longi) 처리
    
    Args:
        lines_by_algo: 알고리즘별 선 검출 결과
        roi_bgr: ROI 이미지
        cls: 클래스 ID
        roi_bbox: ROI 경계 (x1, y1, x2, y2)
        image_width: 전체 이미지 너비
        params: 파라미터 dict (BO에서 전달)
    
    Returns:
        result dict or None
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox
    roi_w, roi_h = roi_bgr.shape[1], roi_bgr.shape[0]
    
    # 1. 방향성 필터
    filtered_after_diag = {}
    for algo, lines in lines_by_algo.items():
        result_dict = filter_lines_by_diagonal(lines, roi_w, roi_h)
        key = 'horizontal' if roi_w > roi_h else 'vertical'
        filtered_after_diag[algo] = result_dict[key]
    
    # 2. 중심 영역 필터
    all_passed_diag_list = [lines for lines in filtered_after_diag.values() if lines.size > 0]
    passed_after_center = np.empty((0, 4), dtype=np.float32)
    
    if all_passed_diag_list:
        all_passed_diag = np.vstack(all_passed_diag_list)
        for tolerance in [0.10, 0.20, 0.30, 0.40, 0.50]:
            passed_after_center = filter_line_by_centrality(
                all_passed_diag, roi_w, roi_h, tolerance_ratio=tolerance
            )
            if passed_after_center.size > 0:
                break
    
    # 3. 가중 RANSAC
    final_candidates = defaultdict(list)
    if passed_after_center.size > 0:
        passed_center_set = {tuple(map(int, row)) for row in passed_after_center}
        for algo, lines in filtered_after_diag.items():
            if lines.size > 0:
                for line in lines:
                    if tuple(map(int, line)) in passed_center_set:
                        final_candidates[algo].append(line)
    
    for algo, lines in final_candidates.items():
        final_candidates[algo] = np.array(lines)

    # ✅ params 사용 (이제 인자로 받음!)
    w_center = float(params.get('ransac_center_w', 0.5))
    w_length = float(params.get('ransac_length_w', 0.5))
    w_consensus = int(params.get('ransac_consensus_w', 5))
    
    ransac_line = weighted_ransac_line(
        final_candidates, roi_w, roi_h,
        w_center=w_center, w_length=w_length,
        consensus_weight=w_consensus
    )
    
    # [디버그] RANSAC 결과 출력
    # print(f"[RANSAC] cand={sum(len(v) for v in final_candidates.values())} | "
    #       f"w_center={w_center:.2f}, w_length={w_length:.2f}, w_consensus={w_consensus} | "
    #       f"success={ransac_line is not None}")

    if ransac_line:
        pt1, pt2 = ransac_line
        pt1_global = (int(pt1[0] + x1_roi), int(pt1[1] + y1_roi))
        pt2_global = (int(pt2[0] + x1_roi), int(pt2[1] + y1_roi))
        return {
            "class_id": cls,
            "type": "ransac_line",
            "line": [pt1_global[0], pt1_global[1], pt2_global[0], pt2_global[1]],
            "roi": roi_bbox,
            "airline_lines_in_roi": lines_by_algo.get("AirLine_Q", np.empty((0, 4)))
        }
    
    return None


def process_collar_roi(lines_by_algo, roi_bgr, cls, roi_bbox, image_width, params):
    """
    클래스 3~6 (collar) 처리
    
    Args:
        params: 일관성을 위해 추가 (현재는 사용 안함)
    
    Returns:
        result dict or None
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi_bbox
    roi_w, roi_h = roi_bgr.shape[1], roi_bgr.shape[0]
    
    is_left = (x1_roi + x2_roi) / 2 < image_width / 2
    three_sides_roi = find_3_sides_of_collar(lines_by_algo, roi_w, roi_h, is_left)
    
    three_sides = {}
    for key, line in three_sides_roi.items():
        if line is not None:
            line_global = [
                line[0] + x1_roi, line[1] + y1_roi,
                line[2] + x1_roi, line[3] + y1_roi
            ]
            three_sides[key] = line_global
    
    if len(three_sides) == 3:
        return {
            "class_id": cls,
            "type": "collar_sides",
            "sides": three_sides,
            "is_left": is_left
        }
    
    return None


def detect_with_full_pipeline(image, params, yolo_detector):
    """
    YOLO + 필터 + 교점 계산 전체 파이프라인
    
    Args:
        image: BGR 이미지
        params: AirLine 파라미터 dict
        yolo_detector: YOLODetector 인스턴스
    
    Returns:
        coords: 12개 좌표 dict
    """
    h, w = image.shape[:2]
    
    # 1. YOLO ROI 검출
    rois = yolo_detector.detect_rois(image)
    
    if not rois:
        print("  [WARN] No ROI detected, processing full image")
        rois = [(0, 0, 0, w, h)]
    
    # 2. ROI별 처리 결과 저장
    processed_results = []
    
    for cls, x1_roi, y1_roi, x2_roi, y2_roi in rois:
        if cls == 0:
            continue
        
        roi_bgr = image[y1_roi:y2_roi, x1_roi:x2_roi]
        if roi_bgr.size == 0:
            continue
        
        # 전처리
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0.8)
        S_roi = sharp_S(roi_gray_blur)
        mean_roi = float(roi_gray_blur.mean())
        
        try:
            roi_bgr_enhanced = enhance_color(roi_bgr, pre_gray=roi_gray_blur, 
                                            pre_S=S_roi, pre_mean=mean_roi)
        except Exception as e:
            print(f"  [WARN] enhance_color failed: {e}")
            roi_bgr_enhanced = roi_bgr.copy()
        
        roi_gray_enhanced = cv2.cvtColor(roi_bgr_enhanced, cv2.COLOR_BGR2GRAY)
        
        # 선 검출
        lines_by_algo = detect_lines_in_roi(roi_bgr_enhanced, roi_gray_enhanced, params)
        
        # 클래스별 처리 (✅ params 전달!)
        if cls in [1, 2]:  # fillet, longi
            result = process_guideline_roi(
                lines_by_algo, roi_bgr, cls, 
                (x1_roi, y1_roi, x2_roi, y2_roi), w, params
            )
            if result:
                processed_results.append(result)
        
        elif cls in [3, 4, 5, 6]:  # collar
            result = process_collar_roi(
                lines_by_algo, roi_bgr, cls,
                (x1_roi, y1_roi, x2_roi, y2_roi), w, params
            )
            if result:
                processed_results.append(result)
    
    # 3. 최종 좌표 계산
    coords = calculate_final_coordinates(processed_results, w, h)
    
    return coords


def calculate_final_coordinates(processed_results, image_width, image_height):
    """
    처리된 결과들로부터 최종 12개 좌표 계산
    
    Returns:
        coords: dict with 12 coordinates
    """
    coords = {
        "longi_left_lower_x": 0, "longi_left_lower_y": 0,
        "longi_right_lower_x": 0, "longi_right_lower_y": 0,
        "longi_left_upper_x": 0, "longi_left_upper_y": 0,
        "longi_right_upper_x": 0, "longi_right_upper_y": 0,
        "collar_left_lower_x": 0, "collar_left_lower_y": 0,
        "collar_left_upper_x": 0, "collar_left_upper_y": 0,
    }
    
    # 결과 분류
    fillet_line = None
    left_longi_res = None
    right_longi_res = None
    collar_sides = None
    
    for res in processed_results:
        if res["class_id"] == 1 and res["type"] == "ransac_line":
            fillet_line = res["line"]
        
        elif res["class_id"] == 2 and res["type"] == "ransac_line":
            roi_center_x = (res["roi"][0] + res["roi"][2]) / 2
            if roi_center_x < image_width / 2:
                left_longi_res = res
            else:
                right_longi_res = res
        
        elif res["type"] == "collar_sides":
            collar_sides = res
    
    # Fillet과 Longi 교점 → Lower Points
    if fillet_line and left_longi_res:
        pt = get_intersection(fillet_line, left_longi_res["line"])
        if pt:
            coords["longi_left_lower_x"], coords["longi_left_lower_y"] = pt
    
    if fillet_line and right_longi_res:
        pt = get_intersection(fillet_line, right_longi_res["line"])
        if pt:
            coords["longi_right_lower_x"], coords["longi_right_lower_y"] = pt
    
    # Upper Points
    if left_longi_res:
        upper_pt_left = find_upper_point_by_intersection(left_longi_res)
        if not upper_pt_left:
            upper_pt_left = find_upper_point_by_roi_fallback(left_longi_res)
        if upper_pt_left:
            coords["longi_left_upper_x"], coords["longi_left_upper_y"] = upper_pt_left
    
    if right_longi_res:
        upper_pt_right = find_upper_point_by_intersection(right_longi_res)
        if not upper_pt_right:
            upper_pt_right = find_upper_point_by_roi_fallback(right_longi_res)
        if upper_pt_right:
            coords["longi_right_upper_x"], coords["longi_right_upper_y"] = upper_pt_right
    
    # Collar Points
    if collar_sides:
        collar_coords = calculate_collar_points(
            collar_sides, left_longi_res, right_longi_res, image_width
        )
        coords.update(collar_coords)
    
    return coords


def calculate_collar_points(collar_sides, left_longi_res, right_longi_res, image_width):
    """Collar 좌표 계산"""
    coords = {
        "collar_left_lower_x": 0, "collar_left_lower_y": 0,
        "collar_left_upper_x": 0, "collar_left_upper_y": 0,
    }
    
    sides = collar_sides["sides"]
    is_left = collar_sides["is_left"]
    
    if len(sides) != 3:
        return coords
    
    longi_to_use = left_longi_res if is_left else right_longi_res
    
    if not longi_to_use:
        return coords
    
    # 4개 교점 계산
    p1 = get_intersection(sides['top'], sides['outer_vertical'])
    p2 = get_intersection(sides['bottom'], sides['outer_vertical'])
    p3 = get_intersection(sides['top'], longi_to_use['line'])
    p4 = get_intersection(sides['bottom'], longi_to_use['line'])
    
    final_points = [p for p in [p1, p2, p3, p4] if p is not None]
    
    if len(final_points) != 4:
        return coords
    
    points = np.array(final_points)
    collar_center_x = np.mean(points[:, 0])
    sorted_by_x = sorted(final_points, key=lambda p: p[0])
    
    if collar_center_x < image_width / 2:
        target_points = sorted_by_x[2:]
    else:
        target_points = sorted_by_x[:2]
    
    if len(target_points) == 2:
        p_a, p_b = target_points
        if p_a[1] < p_b[1]:
            coords["collar_left_upper_x"], coords["collar_left_upper_y"] = int(p_a[0]), int(p_a[1])
            coords["collar_left_lower_x"], coords["collar_left_lower_y"] = int(p_b[0]), int(p_b[1])
        else:
            coords["collar_left_upper_x"], coords["collar_left_upper_y"] = int(p_b[0]), int(p_b[1])
            coords["collar_left_lower_x"], coords["collar_left_lower_y"] = int(p_a[0]), int(p_a[1])
    
    return coords


if __name__ == "__main__":
    # 테스트 코드
    print(f"YOLO_AirLine path: {YOLO_PATH}")
    print(f"Path exists: {YOLO_PATH.exists()}")
    
    try:
        yolo_detector = YOLODetector("models/best.pt")
        
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_params = {
            'edgeThresh1': -3.0,
            'simThresh1': 0.98,
            'pixelRatio1': 0.05,
            'edgeThresh2': 1.0,
            'simThresh2': 0.75,
            'pixelRatio2': 0.05,
        }
        
        result = detect_with_full_pipeline(test_img, test_params, yolo_detector)
        print("✓ Full pipeline test successful!")
        print(f"Result coordinates: {sum(1 for v in result.values() if v != 0)}/12 filled")
        
    except Exception as e:
        print(f"✗ Error: {e}")