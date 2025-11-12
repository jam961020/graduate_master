"""
자동 라벨링 스크립트
AirLine_assemble_test.py를 사용하여 6개 점(longi 4개 + collar 2개)을 자동 추출
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# AirLine_assemble_test import
sys.path.insert(0, str(Path(__file__).parent.parent))
from AirLine_assemble_test import (
    run_airline, run_lsd, run_fld, run_hough,
    enhance_color, get_line_pixels, extend_line,
    filter_lines_by_diagonal, filter_line_by_centrality,
    find_best_fit_line_ransac, get_intersection, sharp_S
)

from yolo_detector import YOLODetector


def auto_label_image(image_path, yolo_detector):
    """
    이미지를 자동으로 라벨링

    Args:
        image_path: 이미지 경로
        yolo_detector: YOLO 검출기

    Returns:
        coords: dict with 12 coordinates (6 points * 2)
        or None if labeling failed
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    h, w = image.shape[:2]

    # 1. YOLO ROI 검출
    try:
        rois = yolo_detector.detect_rois(image)
        if not rois:
            print(f"No ROIs detected for {image_path}")
            return None
    except Exception as e:
        print(f"YOLO detection failed for {image_path}: {e}")
        return None

    # 2. ROI 분류
    guideline_roi = None
    collar_roi = None

    for roi in rois:
        x1, y1, x2, y2, conf, cls = roi
        if int(cls) == 0:  # guideline
            guideline_roi = (int(x1), int(y1), int(x2), int(y2))
        elif int(cls) == 1:  # collar
            collar_roi = (int(x1), int(y1), int(x2), int(y2))

    if guideline_roi is None or collar_roi is None:
        print(f"Missing ROI for {image_path}")
        return None

    # 3. Guideline 처리 (longi left, longi right)
    x1, y1, x2, y2 = guideline_roi
    roi_bgr = image[y1:y2, x1:x2].copy()
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 전처리
    roi_gray_sharp = sharp_S(roi_gray.copy())
    roi_bgr_enhanced = enhance_color(roi_bgr.copy())
    roi_gray_enhanced = cv2.cvtColor(roi_bgr_enhanced, cv2.COLOR_BGR2GRAY)

    # 선 검출 (기본 파라미터 사용)
    lines_airline_Q = run_airline(roi_gray_enhanced, 'Q')
    lines_airline_QG = run_airline(roi_gray_enhanced, 'QG')
    lines_lsd = run_lsd(roi_gray_enhanced)
    lines_fld = run_fld(roi_gray_enhanced)
    lines_hough = run_hough(roi_gray_enhanced)

    lines_by_algo = {
        'AirLine_Q': lines_airline_Q,
        'AirLine_QG': lines_airline_QG,
        'LSD': lines_lsd,
        'FLD': lines_fld,
        'Hough': lines_hough
    }

    # RANSAC으로 최적 직선 찾기
    try:
        longi_line_img = find_best_fit_line_ransac(lines_by_algo, roi_bgr, roi_gray)
        if longi_line_img is None:
            print(f"Failed to find longi line for {image_path}")
            return None

        # 교점 계산으로 4개 점 추출 (left lower, right lower, left upper, right upper)
        # longi_line_img는 ((x1, y1), (x2, y2)) 형식
        pt1, pt2 = longi_line_img

        # ROI 좌표를 원본 이미지 좌표로 변환
        longi_left_lower = (pt1[0] + x1, pt1[1] + y1)
        longi_right_lower = (pt2[0] + x1, pt2[1] + y1)

        # Upper 점은 선을 연장해서 상단 경계와의 교점
        longi_left_upper = (pt1[0] + x1, 0)  # 임시
        longi_right_upper = (pt2[0] + x1, 0)  # 임시

    except Exception as e:
        print(f"Failed to process guideline for {image_path}: {e}")
        return None

    # 4. Collar 처리
    x1, y1, x2, y2 = collar_roi
    roi_bgr = image[y1:y2, x1:x2].copy()
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 전처리
    roi_gray_sharp = sharp_S(roi_gray.copy())
    roi_bgr_enhanced = enhance_color(roi_bgr.copy())
    roi_gray_enhanced = cv2.cvtColor(roi_bgr_enhanced, cv2.COLOR_BGR2GRAY)

    # 선 검출
    lines_airline_Q = run_airline(roi_gray_enhanced, 'Q')
    lines_airline_QG = run_airline(roi_gray_enhanced, 'QG')
    lines_lsd = run_lsd(roi_gray_enhanced)
    lines_fld = run_fld(roi_gray_enhanced)
    lines_hough = run_hough(roi_gray_enhanced)

    lines_by_algo = {
        'AirLine_Q': lines_airline_Q,
        'AirLine_QG': lines_airline_QG,
        'LSD': lines_lsd,
        'FLD': lines_fld,
        'Hough': lines_hough
    }

    # RANSAC으로 최적 직선 찾기
    try:
        collar_line_img = find_best_fit_line_ransac(lines_by_algo, roi_bgr, roi_gray)
        if collar_line_img is None:
            print(f"Failed to find collar line for {image_path}")
            return None

        # 2개 점 추출
        pt1, pt2 = collar_line_img
        collar_left_lower = (pt1[0] + x1, pt1[1] + y1)
        collar_left_upper = (pt2[0] + x1, pt2[1] + y1)

    except Exception as e:
        print(f"Failed to process collar for {image_path}: {e}")
        return None

    # 5. 결과 정리
    coords = {
        "longi_left_lower_x": int(longi_left_lower[0]),
        "longi_left_lower_y": int(longi_left_lower[1]),
        "longi_right_lower_x": int(longi_right_lower[0]),
        "longi_right_lower_y": int(longi_right_lower[1]),
        "longi_left_upper_x": int(longi_left_upper[0]),
        "longi_left_upper_y": int(longi_left_upper[1]),
        "longi_right_upper_x": int(longi_right_upper[0]),
        "longi_right_upper_y": int(longi_right_upper[1]),
        "collar_left_lower_x": int(collar_left_lower[0]),
        "collar_left_lower_y": int(collar_left_lower[1]),
        "collar_left_upper_x": int(collar_left_upper[0]),
        "collar_left_upper_y": int(collar_left_upper[1])
    }

    return coords


def main():
    parser = argparse.ArgumentParser(description="자동 라벨링 스크립트")
    parser.add_argument("--image_dir", type=str, default="../dataset/images/test",
                       help="이미지 디렉토리")
    parser.add_argument("--output", type=str, default="../dataset/ground_truth_auto.json",
                       help="출력 JSON 파일")
    parser.add_argument("--yolo_model", type=str, default="models/best.pt",
                       help="YOLO 모델 경로")
    args = parser.parse_args()

    # YOLO 검출기 초기화
    print("YOLO 검출기 초기화 중...")
    yolo_detector = YOLODetector(args.yolo_model)

    # 이미지 로드
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"Found {len(image_files)} images")

    # 자동 라벨링 실행
    results = {}
    success_count = 0

    for image_path in tqdm(image_files, desc="Auto-labeling"):
        coords = auto_label_image(image_path, yolo_detector)
        if coords is not None:
            results[image_path.name] = {"coordinates": coords}
            success_count += 1
        else:
            print(f"  Failed: {image_path.name}")

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n자동 라벨링 완료!")
    print(f"성공: {success_count}/{len(image_files)}")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
