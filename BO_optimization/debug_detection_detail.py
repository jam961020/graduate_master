# debug_detection_detail.py
"""
검출 과정 상세 디버그
"""
import cv2
import json
import numpy as np
from pathlib import Path
import sys

# 경로 추가
sys.path.insert(0, "BO_optimization")
sys.path.insert(0, "YOLO_AirLine")

# 첫 번째 이미지
with open("dataset/ground_truth.json", 'r') as f:
    gt_data = json.load(f)

first_img = list(gt_data.keys())[0]
img_path = gt_data[first_img]["image"]

image = cv2.imread(img_path)
h, w = image.shape[:2]

print(f"이미지: {first_img}")
print(f"크기: {w}x{h}\n")

# 파라미터
params = {
    'edgeThresh1': -5.0,
    'simThresh1': 0.85,
    'pixelRatio1': 0.05,
    'edgeThresh2': 0.0,
    'simThresh2': 0.80,
    'pixelRatio2': 0.05,
    'weight_both_air': 5.0,
    'weight_one_air': 3.0,
    'weight_overlap': 5.0,
}

# ===== detect_single_line_simple 내부 재현 =====

try:
    from test_clone import run_airline, run_lsd, run_fld, run_hough, enhance_color, get_line_pixels, extend_line
    print("[INFO] Imported from test_clone.py\n")
except ImportError:
    try:
        from AirLine_assemble_test import run_airline, run_lsd, run_fld, run_hough, enhance_color, get_line_pixels, extend_line
        print("[INFO] Imported from AirLine_assemble_test.py\n")
    except ImportError as e:
        print(f"[ERROR] Cannot import: {e}")
        print("\nYOLO_AirLine 폴더의 파일명을 확인하세요:")
        import os
        if os.path.exists("YOLO_AirLine"):
            files = [f for f in os.listdir("YOLO_AirLine") if f.endswith('.py')]
            print("  Available .py files:")
            for f in files:
                print(f"    - {f}")
        exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.8)

try:
    enhanced = enhance_color(image)
except:
    enhanced = image.copy()

diagonal = np.sqrt(h**2 + w**2)

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

print("="*60)
print("검출 과정 디버그")
print("="*60)

print("\n1. AirLine 실행...")
lines_a1 = run_airline(enhanced, config1)
lines_a2 = run_airline(enhanced, config2)

print(f"   AirLine1: {len(lines_a1) if lines_a1 is not None else 0}개 선")
print(f"   AirLine2: {len(lines_a2) if lines_a2 is not None else 0}개 선")

gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

print("\n2. 전통 기법 실행...")
lines_lsd = run_lsd(gray_enh)
lines_fld = run_fld(gray_enh)
lines_hough = run_hough(gray_enh)

print(f"   LSD:   {len(lines_lsd) if lines_lsd is not None else 0}개 선")
print(f"   FLD:   {len(lines_fld) if lines_fld is not None else 0}개 선")
print(f"   Hough: {len(lines_hough) if lines_hough is not None else 0}개 선")

# 통합
all_lines = []
for lines in [lines_a1, lines_a2, lines_lsd, lines_fld, lines_hough]:
    if lines is not None and lines.size > 0:
        all_lines.append(lines)

if len(all_lines) == 0:
    print("\n[ERROR] 검출된 선이 없습니다!")
    exit(1)

all_lines = np.vstack(all_lines)
print(f"\n3. 총 선: {len(all_lines)}개")

# 픽셀 수집
print("\n4. 픽셀 수집 중...")
pixels = []

print("   처음 5개 선:")
for i, line in enumerate(all_lines[:5]):
    x1, y1, x2, y2 = line
    print(f"   Line {i}: ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})")
    try:
        line_pixels = get_line_pixels(int(x1), int(y1), int(x2), int(y2))
        pixels.extend(line_pixels)
    except Exception as e:
        print(f"      [WARN] get_line_pixels failed: {e}")

# 모든 선에서 픽셀 수집
for line in all_lines[5:]:
    x1, y1, x2, y2 = line
    try:
        line_pixels = get_line_pixels(int(x1), int(y1), int(x2), int(y2))
        pixels.extend(line_pixels)
    except:
        pass

pixels = np.array(list(set(map(tuple, pixels))))
print(f"\n   총 수집 픽셀: {len(pixels):,}개")

if len(pixels) < 2:
    print("[ERROR] 픽셀이 너무 적습니다!")
    exit(1)

# 픽셀 범위 확인
print(f"\n5. 픽셀 분포:")
print(f"   X: {pixels[:, 0].min():.0f} ~ {pixels[:, 0].max():.0f} (범위: {np.ptp(pixels[:, 0]):.0f})")
print(f"   Y: {pixels[:, 1].min():.0f} ~ {pixels[:, 1].max():.0f} (범위: {np.ptp(pixels[:, 1]):.0f})")

# 방향 판단
x_range = np.ptp(pixels[:, 0])
y_range = np.ptp(pixels[:, 1])
is_vertical = x_range < y_range

print(f"\n6. 방향 판정:")
print(f"   X 범위: {x_range:.0f} 픽셀")
print(f"   Y 범위: {y_range:.0f} 픽셀")
print(f"   → {'세로선 (X < Y)' if is_vertical else '가로선 (X >= Y)'}")

# RANSAC
print("\n7. RANSAC 실행...")
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(max_trials=1000, random_state=42, min_samples=2)

if is_vertical:
    X = pixels[:, 1].reshape(-1, 1)  # Y를 입력
    y_pred = pixels[:, 0]             # X를 예측
    print("   모델: X = m*Y + c")
else:
    X = pixels[:, 0].reshape(-1, 1)  # X를 입력
    y_pred = pixels[:, 1]             # Y를 예측
    print("   모델: Y = m*X + c")

ransac.fit(X, y_pred)

m = ransac.estimator_.coef_[0]
c = ransac.estimator_.intercept_

print(f"   기울기(m): {m:.6f}")
print(f"   절편(c):   {c:.2f}")

# 선 생성
print("\n8. 선 생성...")
if is_vertical:
    x1, x2 = m * 0 + c, m * h + c
    y1, y2 = 0, h
    print(f"   X = {m:.6f} * Y + {c:.2f}")
    print(f"   Y=0일 때 X={x1:.0f}, Y={h}일 때 X={x2:.0f}")
else:
    y1, y2 = m * 0 + c, m * w + c
    x1, x2 = 0, w
    print(f"   Y = {m:.6f} * X + {c:.2f}")
    print(f"   X=0일 때 Y={y1:.0f}, X={w}일 때 Y={y2:.0f}")

print(f"\n   생성된 선: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")

# extend_line
print("\n9. 선 연장 (extend_line)...")
line_extended = extend_line((x1, y1), (x2, y2), w, h)

if line_extended:
    if isinstance(line_extended[0], tuple):
        (ex1, ey1), (ex2, ey2) = line_extended
    else:
        ex1, ey1, ex2, ey2 = line_extended
    
    print(f"   연장된 선: ({ex1:.0f}, {ey1:.0f}) → ({ex2:.0f}, {ey2:.0f})")
    
    # line_to_coords 적용
    x_avg = int((ex1 + ex2) / 2)
    y_avg = int((ey1 + ey2) / 2)
    
    print(f"\n10. line_to_coords 변환:")
    print(f"   X 평균: {x_avg}")
    print(f"   Y 평균: {y_avg}")
    print(f"   방향: {'세로선' if is_vertical else '가로선'}")
    
    if is_vertical:
        print(f"\n   최종 좌표 (세로선):")
        print(f"     longi_left_lower_x: {x_avg}")
        print(f"     longi_left_lower_y: {max(ey1, ey2):.0f}")
        print(f"     longi_left_upper_x: {x_avg}")
        print(f"     longi_left_upper_y: {min(ey1, ey2):.0f}")
    else:
        print(f"\n   최종 좌표 (가로선):")
        print(f"     longi_left_lower_x: {min(ex1, ex2):.0f}")
        print(f"     longi_left_lower_y: {y_avg}")
        print(f"     longi_right_lower_x: {max(ex1, ex2):.0f}")
        print(f"     longi_right_lower_y: {y_avg}")
else:
    print("   [ERROR] extend_line 실패!")

print("\n" + "="*60)
print("디버그 완료")
print("="*60)