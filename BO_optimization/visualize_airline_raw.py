#!/usr/bin/env python3
"""
AirLine Raw 결과 시각화
ROI별로 Q/QG 프리셋 결과를 각각 표시
"""
import json
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "YOLO_AirLine"))
from AirLine_assemble_test import run_airline, enhance_color, sharp_S

from yolo_detector import YOLODetector


def visualize_airline_raw(image_path, params, yolo_detector, save_dir):
    """
    AirLine Q/QG raw 결과 시각화
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]

    # YOLO ROI 검출
    rois = yolo_detector.detect_rois(img)
    if not rois:
        rois = [(0, 0, 0, w, h)]

    roi_idx = 0
    for cls, x1_roi, y1_roi, x2_roi, y2_roi in rois:
        if cls == 0:
            continue

        roi_bgr = img[y1_roi:y2_roi, x1_roi:x2_roi]
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
        except:
            roi_bgr_enhanced = roi_bgr.copy()

        roi_gray_enhanced = cv2.cvtColor(roi_bgr_enhanced, cv2.COLOR_BGR2GRAY)

        # AirLine Q
        lines_Q = run_airline(
            roi_gray_enhanced,
            edge=params['edgeThresh1'],
            sim=params['simThresh1'],
            pix_ratio=params['pixelRatio1']
        )

        # AirLine QG
        lines_QG = run_airline(
            roi_gray_enhanced,
            edge=params['edgeThresh2'],
            sim=params['simThresh2'],
            pix_ratio=params['pixelRatio2']
        )

        # 시각화 (2개 이미지: Q, QG)
        vis_Q = roi_bgr.copy()
        vis_QG = roi_bgr.copy()

        # Q 선 그리기 (초록)
        if lines_Q is not None and len(lines_Q) > 0:
            for line in lines_Q:
                x1, y1, x2, y2 = line
                cv2.line(vis_Q, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), 2)

        # QG 선 그리기 (빨강)
        if lines_QG is not None and len(lines_QG) > 0:
            for line in lines_QG:
                x1, y1, x2, y2 = line
                cv2.line(vis_QG, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 0, 255), 2)

        # 정보 텍스트
        cv2.putText(vis_Q, f"AirLine Q (cls={cls})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_Q, f"edge={params['edgeThresh1']:.2f}, sim={params['simThresh1']:.3f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_Q, f"{len(lines_Q) if lines_Q else 0} lines", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(vis_QG, f"AirLine QG (cls={cls})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_QG, f"edge={params['edgeThresh2']:.2f}, sim={params['simThresh2']:.3f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_QG, f"{len(lines_QG) if lines_QG else 0} lines", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 저장
        save_Q = save_dir / f"roi{roi_idx:02d}_cls{cls}_Q.png"
        save_QG = save_dir / f"roi{roi_idx:02d}_cls{cls}_QG.png"
        cv2.imwrite(str(save_Q), vis_Q)
        cv2.imwrite(str(save_QG), vis_QG)

        print(f"    ROI {roi_idx} (cls={cls}): Q={len(lines_Q) if lines_Q else 0} lines, QG={len(lines_QG) if lines_QG else 0} lines")

        roi_idx += 1


def main(log_dir):
    """Best iteration의 AirLine raw 결과 시각화"""
    log_path = Path(log_dir)
    session_name = log_path.name

    print("="*70)
    print(f"AIRLINE RAW VISUALIZATION - {session_name}")
    print("="*70)

    # Best iteration 찾기
    iter_files = sorted(log_path.glob("iter_*.json"),
                       key=lambda x: int(x.stem.split('_')[1]))

    iterations = []
    for f in iter_files:
        with open(f) as file:
            d = json.load(file)
            iterations.append(d)

    best = max(iterations, key=lambda x: x.get('cvar', 0))
    print(f"\nBest iteration: {best['iteration']} (CVaR={best['cvar']:.4f})")

    # 이미지 로드
    gt_file = Path(__file__).parent.parent / "dataset" / "ground_truth.json"
    with open(gt_file) as f:
        gt_data = json.load(f)

    image_dir = Path(__file__).parent.parent / "dataset" / "images" / "test"
    all_images = []
    for img_name in gt_data.keys():
        for ext in ['.jpg', '.png']:
            p = image_dir / f"{img_name}{ext}"
            if p.exists():
                all_images.append(p)
                break

    image_idx = best['image_idx']
    img_path = all_images[image_idx]
    params = best['parameters']

    print(f"Image: {img_path.name}\n")

    # YOLO 초기화
    yolo_model_path = Path(__file__).parent / "models" / "best.pt"
    yolo_detector = YOLODetector(str(yolo_model_path))

    # 저장 디렉토리
    output_dir = Path("results") / f"{session_name}_airline_raw" / f"iter{best['iteration']:03d}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 시각화
    visualize_airline_raw(img_path, params, yolo_detector, output_dir)

    print(f"\n✓ Saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_airline_raw.py <log_dir>")
        print("\nExample:")
        print("  python visualize_airline_raw.py logs/run_20251114_172045")
        sys.exit(1)

    main(sys.argv[1])
