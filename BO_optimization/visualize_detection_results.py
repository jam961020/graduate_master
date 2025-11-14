#!/usr/bin/env python3
"""
검출 결과 시각화 스크립트
Session 13의 Best/Worst CVaR iteration에서 실제로 어떤 선이 검출되는지 확인
"""
import json
import cv2
import numpy as np
from pathlib import Path
import sys

# full_pipeline import
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector

def load_iteration_data(log_dir, iteration):
    """특정 iteration 데이터 로드"""
    iter_file = Path(log_dir) / f"iter_{iteration:03d}.json"
    if not iter_file.exists():
        print(f"Error: {iter_file} not found")
        return None

    with open(iter_file) as f:
        return json.load(f)

def visualize_detection(image_path, params, yolo_detector, gt_coords=None, save_path=None):
    """
    이미지에서 검출 결과 시각화

    Args:
        image_path: 이미지 경로
        params: 파라미터 딕셔너리 (edgeThresh1, simThresh1, ...)
        yolo_detector: YOLO 검출기
        gt_coords: Ground truth 좌표 dict (optional)
        save_path: 저장 경로 (optional)

    Returns:
        vis_img: 시각화된 이미지
    """
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot load {image_path}")
        return None

    h, w = img.shape[:2]
    vis_img = img.copy()

    # RANSAC 가중치 추출 (파라미터에 있으면 사용, 없으면 기본값)
    ransac_weights = (
        params.get('ransac_weight_q', 5),
        params.get('ransac_weight_qg', 5)
    )

    # 전체 파이프라인 실행
    detected_coords = detect_with_full_pipeline(img, params, yolo_detector, ransac_weights)

    # 검출된 좌표 그리기 (longi guideline)
    def draw_line_from_coords(img, coords, color, thickness=2):
        """좌표 dict에서 선 그리기"""
        # Longi guideline (lower, upper)
        longi_pts = []
        for pos in ['lower', 'upper']:
            for side in ['left', 'right']:
                x_key = f'longi_{side}_{pos}_x'
                y_key = f'longi_{side}_{pos}_y'
                x = coords.get(x_key, 0)
                y = coords.get(y_key, 0)
                if x > 0 and y > 0:
                    longi_pts.append((int(x), int(y)))

        if len(longi_pts) >= 2:
            for i in range(len(longi_pts)-1):
                cv2.line(img, longi_pts[i], longi_pts[i+1], color, thickness)
            # 점도 표시
            for pt in longi_pts:
                cv2.circle(img, pt, 5, color, -1)

        # Collar도 그릴 수 있지만 생략
        return img

    # 검출된 선 그리기 (빨간색)
    vis_img = draw_line_from_coords(vis_img, detected_coords, (0, 0, 255), 3)

    # Ground truth 그리기 (초록색)
    if gt_coords is not None:
        vis_img = draw_line_from_coords(vis_img, gt_coords, (0, 255, 0), 2)

    # 정보 텍스트
    info_text = f"ET1={params.get('edgeThresh1', 0):.2f}, ST1={params.get('simThresh1', 0):.3f}"
    cv2.putText(vis_img, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 검출 성공 여부
    longi_detected = any(detected_coords.get(f'longi_left_lower_{ax}', 0) > 0 for ax in ['x', 'y'])
    status = "DETECTED" if longi_detected else "FAILED"
    color = (0, 255, 0) if longi_detected else (0, 0, 255)
    cv2.putText(vis_img, status, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if save_path:
        cv2.imwrite(str(save_path), vis_img)
        print(f"  Saved: {save_path}")

    return vis_img

def main(log_dir, output_dir, visualize_all=False):
    """
    Session 13의 검출 결과 시각화
    - 각 iteration에서 **실제로 평가한 이미지**만 시각화

    Args:
        log_dir: logs/run_YYYYMMDD_HHMMSS
        output_dir: 저장 디렉토리
        visualize_all: True면 모든 iteration, False면 Best/Median/Worst만
    """
    log_path = Path(log_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 모든 iteration 데이터 로드
    iter_files = sorted(log_path.glob("iter_*.json"),
                       key=lambda x: int(x.stem.split('_')[1]))

    # CVaR 수집 + 이미지 인덱스 저장
    iterations_data = []
    for f in iter_files:
        with open(f) as file:
            d = json.load(file)
            iterations_data.append({
                'iteration': d['iteration'],
                'cvar': d.get('cvar', 0),
                'score': d.get('score', 0),
                'image_idx': d.get('image_idx'),
                'params': d.get('parameters', {})
            })

    # CVaR로 정렬
    sorted_by_cvar = sorted(iterations_data, key=lambda x: x['cvar'], reverse=True)

    # Best, Median, Worst 선택
    best = sorted_by_cvar[0]
    median = sorted_by_cvar[len(sorted_by_cvar)//2]
    worst = sorted_by_cvar[-1]

    print("="*70)
    print("DETECTION VISUALIZATION - Actual Evaluated Images Only")
    print("="*70)
    print(f"Best CVaR:   Iter {best['iteration']:3d} | CVaR={best['cvar']:.4f} | Score={best['score']:.4f} | Image #{best['image_idx']}")
    print(f"Median CVaR: Iter {median['iteration']:3d} | CVaR={median['cvar']:.4f} | Score={median['score']:.4f} | Image #{median['image_idx']}")
    print(f"Worst CVaR:  Iter {worst['iteration']:3d} | CVaR={worst['cvar']:.4f} | Score={worst['score']:.4f} | Image #{worst['image_idx']}")
    print("="*70)

    # GT 로드
    gt_file = Path(__file__).parent.parent / "dataset" / "ground_truth.json"
    if not gt_file.exists():
        print(f"Error: GT file not found: {gt_file}")
        return

    with open(gt_file) as f:
        gt_data = json.load(f)

    # 이미지 로드 (GT 파일 순서대로, optimization.py와 동일)
    image_dir = Path(__file__).parent.parent / "dataset" / "images" / "test"
    all_images = []
    for img_name in gt_data.keys():
        possible_paths = [
            image_dir / f"{img_name}.jpg",
            image_dir / f"{img_name}.png",
        ]
        for p in possible_paths:
            if p.exists():
                all_images.append(p)
                break

    print(f"Loaded {len(all_images)} images from GT file order")

    # YOLO detector 초기화
    print("\nInitializing YOLO detector...")
    yolo_model_path = Path(__file__).parent / "models" / "best.pt"
    if not yolo_model_path.exists():
        print(f"Error: YOLO model not found: {yolo_model_path}")
        return
    yolo_detector = YOLODetector(str(yolo_model_path))

    # 시각화할 iteration 선택
    if visualize_all:
        to_visualize = iterations_data
        print(f"\nVisualizing ALL {len(to_visualize)} iterations...")
    else:
        to_visualize = [best, median, worst]
        print(f"\nVisualizing Best/Median/Worst (3 iterations)...")

    # 시각화 실행
    for data in to_visualize:
        iter_num = data['iteration']
        image_idx = data['image_idx']
        params = data['params']
        cvar = data['cvar']
        score = data['score']

        if image_idx is None or image_idx >= len(all_images):
            print(f"\nIter {iter_num}: Invalid image_idx={image_idx}, skipping...")
            continue

        # 실제로 평가한 이미지
        img_path = all_images[image_idx]
        img_name = img_path.name
        gt_coords = gt_data.get(img_name, {}).get('coordinates', gt_data.get(img_name, {}))

        save_path = output_path / f"iter{iter_num:03d}_cvar{cvar:.4f}_score{score:.4f}_{img_name}"

        print(f"\nIter {iter_num:3d} (CVaR={cvar:.4f}, Score={score:.4f}):")
        print(f"  Image #{image_idx}: {img_name}")
        visualize_detection(img_path, params, yolo_detector, gt_coords, save_path)

    print("\n" + "="*70)
    print(f"Visualization complete! Check: {output_path}")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_detection_results.py <log_dir> [output_dir] [--all]")
        print("\nExamples:")
        print("  # Best/Median/Worst만 시각화 (빠름)")
        print("  python visualize_detection_results.py logs/run_20251114_172045")
        print("")
        print("  # 모든 iteration 시각화 (느림, 115개)")
        print("  python visualize_detection_results.py logs/run_20251114_172045 detection_vis --all")
        sys.exit(1)

    log_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '--all' else "detection_visualization"
    visualize_all = '--all' in sys.argv

    main(log_dir, output_dir, visualize_all)
