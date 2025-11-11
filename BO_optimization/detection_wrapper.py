"""
기존 AirLine_assemble_test.py를 BO용으로 래핑
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# 기존 YOLO_AirLine 경로 추가
YOLO_PATH = Path(__file__).parent.parent / "YOLO_AirLine"
sys.path.insert(0, str(YOLO_PATH))

# 이제 직접 import (파일명에서 .py 제거)
try:
    # 방법 1: 파일명이 test_clone.py인 경우
    from test_clone import (
        run_airline, run_lsd, run_fld, run_hough,
        enhance_color, get_line_pixels, extend_line
    )
    print("[INFO] Imported from test_clone.py")
except ImportError:
    try:
        # 방법 2: 파일명이 AirLine_assemble_test.py인 경우
        from AirLine_assemble_test import (
            run_airline, run_lsd, run_fld, run_hough,
            enhance_color, get_line_pixels, extend_line
        )
        print("[INFO] Imported from AirLine_assemble_test.py")
    except ImportError as e:
        print(f"[ERROR] Failed to import from YOLO_AirLine: {e}")
        print(f"[ERROR] Path checked: {YOLO_PATH}")
        print(f"[ERROR] Files in directory:")
        if YOLO_PATH.exists():
            for f in YOLO_PATH.glob("*.py"):
                print(f"  - {f.name}")
        else:
            print(f"  Directory not found: {YOLO_PATH}")
        raise


def detect_single_line_simple(image, params):
    """
    단순화된 검출: ROI 없이 전체 이미지에서 한 선만
    
    Args:
        image: BGR 이미지
        params: AirLine 파라미터 dict
    
    Returns:
        coords: 좌표 dict (8개 값)
    """
    h, w = image.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    
    # 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
    
    try:
        enhanced = enhance_color(image)
    except Exception as e:
        print(f"[WARN] enhance_color failed: {e}, using original")
        enhanced = image.copy()
    
    # AirLine 두 번
    config1 = {
        "edgeThresh": params['edgeThresh1'],
        "simThresh": params['simThresh1'],
        "pixelNumThresh": int(diagonal * params['pixelRatio1'])
    }
    config2 = {
        "edgeThresh": params['edgeThresh2'],
        "simThresh": params['simThresh2'],
        "pixelNumThresh": int(diagonal * params['pixelRatio2'])
    }
    
    try:
        lines_a1 = run_airline(enhanced, config1)
        lines_a2 = run_airline(enhanced, config2)
    except Exception as e:
        print(f"[ERROR] AirLine failed: {e}")
        return empty_coords()
    
    # 전통 기법
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    try:
        lines_lsd = run_lsd(gray_enh)
        lines_fld = run_fld(gray_enh)
        lines_hough = run_hough(gray_enh)
    except Exception as e:
        print(f"[WARN] Traditional methods failed: {e}")
        lines_lsd = np.empty((0, 4))
        lines_fld = np.empty((0, 4))
        lines_hough = np.empty((0, 4))
    
    # 모든 선 통합
    all_lines = []
    for lines in [lines_a1, lines_a2, lines_lsd, lines_fld, lines_hough]:
        if lines is not None and lines.size > 0:
            all_lines.append(lines)
    
    if len(all_lines) == 0:
        return empty_coords()
    
    all_lines = np.vstack(all_lines)
    
    # 픽셀 수집
    pixels = []
    for line in all_lines:
        x1, y1, x2, y2 = line
        try:
            line_pixels = get_line_pixels(int(x1), int(y1), int(x2), int(y2))
            pixels.extend(line_pixels)
        except Exception as e:
            continue
    
    if len(pixels) < 2:
        return empty_coords()
    
    # 중복 제거
    pixels = np.array(list(set(map(tuple, pixels))))
    
    # RANSAC
    from sklearn.linear_model import RANSACRegressor
    ransac = RANSACRegressor(max_trials=1000, random_state=42, min_samples=2)
    
    try:
        x_range = np.ptp(pixels[:, 0])
        y_range = np.ptp(pixels[:, 1])
        is_vertical = x_range < y_range
        
        if is_vertical:
            X = pixels[:, 1].reshape(-1, 1)
            y = pixels[:, 0]
        else:
            X = pixels[:, 0].reshape(-1, 1)
            y = pixels[:, 1]
        
        ransac.fit(X, y)
        
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_
        
        if is_vertical:
            x1, x2 = m * 0 + c, m * h + c
            y1, y2 = 0, h
        else:
            y1, y2 = m * 0 + c, m * w + c
            x1, x2 = 0, w
        
        # 연장
        line = extend_line((x1, y1), (x2, y2), w, h)
        
        if line:
            return line_to_coords(line, is_vertical)
        
    except Exception as e:
        print(f"[ERROR] RANSAC failed: {e}")
    
    return empty_coords()


def empty_coords():
    """빈 좌표 dict"""
    return {
        "longi_left_lower_x": 0, "longi_left_lower_y": 0,
        "longi_right_lower_x": 0, "longi_right_lower_y": 0,
        "longi_left_upper_x": 0, "longi_left_upper_y": 0,
        "longi_right_upper_x": 0, "longi_right_upper_y": 0,
        "collar_left_lower_x": 0, "collar_left_lower_y": 0,
        "collar_left_upper_x": 0, "collar_left_upper_y": 0,
    }


def line_to_coords(line, is_vertical):
    """
    선을 좌표로 변환 - 경계값 대신 실제 값 사용
    """
    if isinstance(line[0], tuple):
        (x1, y1), (x2, y2) = line
    else:
        x1, y1, x2, y2 = line
    
    coords = empty_coords()
    
    if is_vertical:
        # 세로선: X는 평균, Y는 상/하
        x_avg = int((x1 + x2) / 2)
        
        coords["longi_left_lower_x"] = x_avg
        coords["longi_left_lower_y"] = int(max(y1, y2))
        coords["longi_left_upper_x"] = x_avg
        coords["longi_left_upper_y"] = int(min(y1, y2))
        
        # Right는 일단 동일 (단일 선이므로)
        coords["longi_right_lower_x"] = x_avg
        coords["longi_right_lower_y"] = int(max(y1, y2))
        coords["longi_right_upper_x"] = x_avg
        coords["longi_right_upper_y"] = int(min(y1, y2))
    else:
        # 가로선: Y는 평균, X는 좌/우
        y_avg = int((y1 + y2) / 2)
        
        coords["longi_left_lower_x"] = int(min(x1, x2))
        coords["longi_left_lower_y"] = y_avg
        coords["longi_left_upper_x"] = int(min(x1, x2))
        coords["longi_left_upper_y"] = y_avg
        
        coords["longi_right_lower_x"] = int(max(x1, x2))
        coords["longi_right_lower_y"] = y_avg
        coords["longi_right_upper_x"] = int(max(x1, x2))
        coords["longi_right_upper_y"] = y_avg
    
    return coords

# 테스트용
if __name__ == "__main__":
    print(f"YOLO_AirLine path: {YOLO_PATH}")
    print(f"Path exists: {YOLO_PATH.exists()}")
    
    if YOLO_PATH.exists():
        print("\nPython files in YOLO_AirLine:")
        for py_file in YOLO_PATH.glob("*.py"):
            print(f"  - {py_file.name}")
    
    print("\nTrying import test...")
    try:
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_params = {
            'edgeThresh1': -3.0,
            'simThresh1': 0.98,
            'pixelRatio1': 0.05,
            'edgeThresh2': 1.0,
            'simThresh2': 0.75,
            'pixelRatio2': 0.05,
        }
        
        result = detect_single_line_simple(test_img, test_params)
        print("✓ Import and function call successful!")
        print(f"Result: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")