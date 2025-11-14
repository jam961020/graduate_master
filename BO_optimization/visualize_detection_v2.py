#!/usr/bin/env python3
"""
검출 결과 시각화 V2
- GT 선 제대로 그리기 (4개 선: longi 양쪽, fillet, collar)
- 검출 결과 선 그리기
- ROI별 AirLine 중간 결과도 저장
- results 폴더에 체계적으로 저장
"""
import json
import cv2
import numpy as np
from pathlib import Path
import sys

from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector


def draw_lines_from_coords(img, coords, color, thickness=2, label_prefix=""):
    """
    좌표 dict에서 4개 선 그리기

    Args:
        img: 이미지
        coords: 좌표 dict (12개)
        color: BGR 색상
        thickness: 선 두께
        label_prefix: 라벨 접두사 ("GT", "Detected" 등)
    """
    vis_img = img.copy()

    # 선 정의 (6개: 세로 3개 + 가로 2개 + collar 세로)
    lines = [
        # 세로선 2개
        ("Longi Left",
         "longi_left_lower_x", "longi_left_lower_y",
         "longi_left_upper_x", "longi_left_upper_y"),
        ("Longi Right",
         "longi_right_lower_x", "longi_right_lower_y",
         "longi_right_upper_x", "longi_right_upper_y"),
        # 가로선 2개 (collar 있을 때)
        ("Fillet Left",
         "longi_left_lower_x", "longi_left_lower_y",
         "collar_left_lower_x", "collar_left_lower_y"),
        ("Fillet Right",
         "collar_left_lower_x", "collar_left_lower_y",
         "longi_right_lower_x", "longi_right_lower_y"),
        # Collar 세로선
        ("Collar Left",
         "collar_left_lower_x", "collar_left_lower_y",
         "collar_left_upper_x", "collar_left_upper_y"),
    ]

    drawn_count = 0

    for line_name, x1_key, y1_key, x2_key, y2_key in lines:
        x1 = coords.get(x1_key, 0)
        y1 = coords.get(y1_key, 0)
        x2 = coords.get(x2_key, 0)
        y2 = coords.get(y2_key, 0)

        # 유효한 좌표인지 확인
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            # 선 그리기
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)),
                    color, thickness)

            # 끝점 표시
            cv2.circle(vis_img, (int(x1), int(y1)), 5, color, -1)
            cv2.circle(vis_img, (int(x2), int(y2)), 5, color, -1)

            # 라벨 (선 중간 지점)
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            label = f"{label_prefix} {line_name}" if label_prefix else line_name
            cv2.putText(vis_img, label, (mid_x + 5, mid_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            drawn_count += 1

    return vis_img, drawn_count


def visualize_full_result(image_path, params, yolo_detector, gt_coords,
                          save_dir, iter_num, cvar, score):
    """
    전체 검출 결과 시각화

    Returns:
        detected_coords: 검출된 좌표 dict
    """
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot load {image_path}")
        return None

    # RANSAC 가중치
    ransac_weights = (
        params.get('ransac_weight_q', 5),
        params.get('ransac_weight_qg', 5)
    )

    # 전체 파이프라인 실행
    detected_coords = detect_with_full_pipeline(img, params, yolo_detector, ransac_weights)

    # 시각화 이미지 생성
    vis_img = img.copy()

    # 1. GT 선 그리기 (초록색)
    if gt_coords:
        vis_img, gt_count = draw_lines_from_coords(vis_img, gt_coords,
                                                    (0, 255, 0), 2, "GT")

    # 2. 검출된 선 그리기 (빨간색)
    vis_img, det_count = draw_lines_from_coords(vis_img, detected_coords,
                                                 (0, 0, 255), 3, "Det")

    # 3. 정보 텍스트
    info_lines = [
        f"Iter {iter_num} | CVaR: {cvar:.4f} | Score: {score:.4f}",
        f"Params: ET1={params.get('edgeThresh1', 0):.2f}, ST1={params.get('simThresh1', 0):.3f}",
        f"        RW_Q={ransac_weights[0]:.2f}, RW_QG={ransac_weights[1]:.2f}",
        f"GT lines: {gt_count if gt_coords else 0} | Detected: {det_count}"
    ]

    y_offset = 25
    for line in info_lines:
        cv2.putText(vis_img, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 25

    # 저장
    save_path = save_dir / "full_result.png"
    cv2.imwrite(str(save_path), vis_img)
    print(f"  Saved: {save_path}")

    return detected_coords


def save_metadata(save_dir, iter_num, cvar, score, params,
                 image_name, gt_coords, detected_coords):
    """
    메타데이터 JSON 저장
    """
    metadata = {
        "iteration": iter_num,
        "cvar": cvar,
        "score": score,
        "image_name": image_name,
        "parameters": params,
        "gt_coords": gt_coords if gt_coords else {},
        "detected_coords": detected_coords
    }

    save_path = save_dir / "metadata.json"
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {save_path}")


def main(log_dir, visualize_all=False):
    """
    Session 검출 결과 시각화

    Args:
        log_dir: logs/run_YYYYMMDD_HHMMSS
        visualize_all: True면 모든 iteration, False면 Best/Median/Worst만
    """
    log_path = Path(log_dir)

    # 세션 이름 추출
    session_name = log_path.name

    # 출력 디렉토리 (results 아래)
    output_base = Path("results") / f"{session_name}_detection"
    output_base.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print(f"DETECTION VISUALIZATION V2 - {session_name}")
    print("="*70)

    # 모든 iteration 데이터 로드
    iter_files = sorted(log_path.glob("iter_*.json"),
                       key=lambda x: int(x.stem.split('_')[1]))

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

    print(f"Best CVaR:   Iter {best['iteration']:3d} | CVaR={best['cvar']:.4f} | Score={best['score']:.4f}")
    print(f"Median CVaR: Iter {median['iteration']:3d} | CVaR={median['cvar']:.4f} | Score={median['score']:.4f}")
    print(f"Worst CVaR:  Iter {worst['iteration']:3d} | CVaR={worst['cvar']:.4f} | Score={worst['score']:.4f}")
    print("="*70)

    # GT 로드
    gt_file = Path(__file__).parent.parent / "dataset" / "ground_truth.json"
    if not gt_file.exists():
        print(f"Error: GT file not found: {gt_file}")
        return

    with open(gt_file) as f:
        gt_data = json.load(f)

    # 이미지 로드 (GT 순서)
    image_dir = Path(__file__).parent.parent / "dataset" / "images" / "test"
    all_images = []
    for img_name in gt_data.keys():
        for ext in ['.jpg', '.png']:
            p = image_dir / f"{img_name}{ext}"
            if p.exists():
                all_images.append(p)
                break

    print(f"Loaded {len(all_images)} images\n")

    # YOLO detector 초기화
    print("Initializing YOLO detector...")
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
        print(f"\nVisualizing Best/Median/Worst (3 iterations)...\n")

    # 시각화 실행
    for data in to_visualize:
        iter_num = data['iteration']
        image_idx = data['image_idx']
        params = data['params']
        cvar = data['cvar']
        score = data['score']

        if image_idx is None or image_idx >= len(all_images):
            print(f"Iter {iter_num}: Invalid image_idx={image_idx}, skipping...")
            continue

        # 실제로 평가한 이미지
        img_path = all_images[image_idx]
        img_name = img_path.name
        gt_coords = gt_data.get(img_name.split('.')[0], {}).get('coordinates', {})

        # 저장 디렉토리 생성
        iter_label = "best" if data == best else ("median" if data == median else "worst")
        save_dir = output_base / f"iter{iter_num:03d}_{iter_label}_cvar{cvar:.4f}"
        save_dir.mkdir(exist_ok=True, parents=True)

        print(f"Iter {iter_num:3d} ({iter_label}) - CVaR={cvar:.4f}, Score={score:.4f}")
        print(f"  Image #{image_idx}: {img_name}")

        # 전체 결과 시각화
        detected_coords = visualize_full_result(
            img_path, params, yolo_detector, gt_coords,
            save_dir, iter_num, cvar, score
        )

        # 메타데이터 저장
        if detected_coords:
            save_metadata(save_dir, iter_num, cvar, score, params,
                         img_name, gt_coords, detected_coords)

        print()

    print("="*70)
    print(f"Visualization complete!")
    print(f"Results saved to: {output_base}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_detection_v2.py <log_dir> [--all]")
        print("\nExamples:")
        print("  # Best/Median/Worst만 시각화")
        print("  python visualize_detection_v2.py logs/run_20251114_172045")
        print("")
        print("  # 모든 iteration 시각화")
        print("  python visualize_detection_v2.py logs/run_20251114_172045 --all")
        sys.exit(1)

    log_dir = sys.argv[1]
    visualize_all = '--all' in sys.argv

    main(log_dir, visualize_all)
