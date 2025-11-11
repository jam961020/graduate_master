# debug_evaluation.py
"""
LP 계산 디버그
"""
import cv2
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "BO_optimization")

from detection_wrapper import detect_single_line_simple
from evaluation import evaluate_lp, evaluate_endpoint_error

# 첫 번째 이미지로 테스트
with open("dataset/ground_truth.json", 'r') as f:
    gt_data = json.load(f)

first_img = list(gt_data.keys())[0]
img_path = gt_data[first_img]["image"]
gt_coords = gt_data[first_img]["coordinates"]

print("="*60)
print(f"테스트 이미지: {first_img}")
print("="*60)

print(f"\nGT 좌표 (픽셀):")
print(f"  longi_left_lower:  ({gt_coords['longi_left_lower_x']}, {gt_coords['longi_left_lower_y']})")
print(f"  longi_right_lower: ({gt_coords['longi_right_lower_x']}, {gt_coords['longi_right_lower_y']})")
print(f"  longi_left_upper:  ({gt_coords['longi_left_upper_x']}, {gt_coords['longi_left_upper_y']})")
print(f"  longi_right_upper: ({gt_coords['longi_right_upper_x']}, {gt_coords['longi_right_upper_y']})")

# 이미지 로드
image = cv2.imread(img_path)
if image is None:
    print(f"[ERROR] Cannot load: {img_path}")
    exit(1)

h, w = image.shape[:2]
print(f"\n이미지 크기: {w}x{h}")

# 테스트 1: GT를 detected로 사용 (완벽한 경우)
print("\n" + "="*60)
print("테스트 1: GT를 검출 결과로 사용 (완벽한 경우)")
print("="*60)

lp_perfect = evaluate_lp(gt_coords, image, first_img)
endpoint_perfect = evaluate_endpoint_error(gt_coords, image, first_img)

print(f"LP Score:       {lp_perfect:.4f} (예상: ~1.0)")
print(f"Endpoint Score: {endpoint_perfect:.4f} (예상: ~1.0)")

# 테스트 2: 실제 검출
print("\n" + "="*60)
print("테스트 2: 실제 검출 결과")
print("="*60)

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

print("검출 중...")
detected_coords = detect_single_line_simple(image, params)

print(f"\n검출된 좌표:")
non_zero = 0
for key, val in detected_coords.items():
    if val != 0:
        print(f"  {key}: {val}")
        non_zero += 1

print(f"\n검출된 좌표 개수: {non_zero}/12")

if non_zero == 0:
    print("\n[ERROR] 모든 좌표가 0입니다!")
    print("검출 실패 원인:")
    print("  1. 이미지가 너무 어둡거나 밝음")
    print("  2. 파라미터가 부적절함")
    print("  3. 선이 실제로 없음")
else:
    # LP 점수 계산
    try:
        lp_score = evaluate_lp(detected_coords, image, first_img)
        endpoint_score = evaluate_endpoint_error(detected_coords, image, first_img)
        
        print(f"\nLP Score:       {lp_score:.4f}")
        print(f"Endpoint Score: {endpoint_score:.4f}")
        
        if lp_score == 0.0 and endpoint_score == 0.0:
            print("\n[WARNING] 점수가 0입니다!")
            print("가능한 원인:")
            print("  1. 검출된 선이 GT와 너무 멀리 떨어짐")
            print("  2. extract_lines_from_coords에서 선 추출 실패")
            
            # 선 추출 디버그
            from evaluation import extract_lines_from_coords
            gt_lines = extract_lines_from_coords(gt_coords)
            det_lines = extract_lines_from_coords(detected_coords)
            
            print(f"\n  GT 선 개수: {len(gt_lines)}")
            print(f"  검출 선 개수: {len(det_lines)}")
            
            if len(det_lines) > 0:
                print(f"\n  GT 첫 번째 선: {gt_lines[0]}")
                print(f"  검출 첫 번째 선: {det_lines[0]}")
    
    except Exception as e:
        print(f"\n[ERROR] 평가 실패: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)