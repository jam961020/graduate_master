"""
ROI 추출 시각화 - 제대로 작동하는지 확인
"""

import cv2
import json
import numpy as np
from pathlib import Path

from yolo_detector import YOLODetector
from environment_independent import extract_parameter_independent_environment
from clip_environment import CLIPEnvironmentEncoder


def visualize_roi_extraction(image_path, yolo_detector, clip_encoder, output_dir):
    """
    ROI 추출 과정 시각화

    Args:
        image_path: 이미지 경로
        yolo_detector: YOLO 검출기
        clip_encoder: CLIP 인코더
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    img_name = Path(image_path).stem
    print(f"\n{'='*70}")
    print(f"Processing: {img_name}")
    print(f"{'='*70}")

    # 1. 원본 이미지 로드
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    print(f"Original image size: {w}x{h}")

    # 2. YOLO ROI 검출
    rois = yolo_detector.detect_rois(image)
    print(f"\nYOLO detected {len(rois)} ROIs:")
    for i, (cls, x1, y1, x2, y2) in enumerate(rois):
        class_name = yolo_detector.CLASS_NAMES[cls]
        print(f"  [{i}] Class {cls} ({class_name}): ({x1}, {y1}) -> ({x2}, {y2})")

    # 3. longi_WL (class 2) 선택
    longi_roi = [roi for roi in rois if roi[0] == 2]

    if longi_roi:
        _, x1, y1, x2, y2 = longi_roi[0]
        print(f"\nSelected: longi_WL (class 2)")
    elif len(rois) > 0:
        _, x1, y1, x2, y2 = rois[0]
        print(f"\nSelected: first ROI (class {rois[0][0]})")
    else:
        print(f"\n⚠️  No ROI detected! Using full image")
        x1, y1, x2, y2 = 0, 0, w, h

    print(f"ROI bbox: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"ROI size: {x2-x1}x{y2-y1}")

    # 4. ROI 크롭
    roi_crop = image[y1:y2, x1:x2]

    # 5. Baseline features 추출
    print(f"\n{'='*70}")
    print("Baseline Features (ROI-based):")
    print(f"{'='*70}")

    roi_bbox = (x1, y1, x2, y2)
    baseline_env = extract_parameter_independent_environment(image, roi=roi_bbox)

    for key, val in baseline_env.items():
        print(f"  {key:<25}: {val:.4f}")

    # 6. CLIP features 추출
    print(f"\n{'='*70}")
    print("CLIP Features (ROI-based):")
    print(f"{'='*70}")

    clip_features = clip_encoder.encode_roi(roi_crop)
    feature_names = clip_encoder.get_feature_names()

    for name, val in zip(feature_names, clip_features):
        print(f"  {name:<25}: {val:.4f}")

    # 7. 시각화 이미지 생성
    print(f"\n{'='*70}")
    print("Generating visualization...")
    print(f"{'='*70}")

    # 원본 + ROI 박스
    vis_full = image.copy()
    cv2.rectangle(vis_full, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.putText(vis_full, f"ROI: {x2-x1}x{y2-y1}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # 크롭된 ROI
    vis_crop = roi_crop.copy()

    # 결합 이미지 (원본 + 크롭)
    # 크기 조정 (원본을 크롭과 비슷한 높이로)
    crop_h = roi_crop.shape[0]
    scale = crop_h / h
    resized_full = cv2.resize(vis_full, (int(w * scale), crop_h))

    combined = np.hstack([resized_full, vis_crop])

    # 저장
    output_combined = output_dir / f"{img_name}_roi_extraction.jpg"
    cv2.imwrite(str(output_combined), combined)
    print(f"  Saved: {output_combined}")

    # 크롭만 따로 저장
    output_crop = output_dir / f"{img_name}_roi_crop.jpg"
    cv2.imwrite(str(output_crop), roi_crop)
    print(f"  Saved: {output_crop}")

    return {
        'image': img_name,
        'roi_bbox': (x1, y1, x2, y2),
        'roi_size': (x2-x1, y2-y1),
        'baseline': baseline_env,
        'clip': {name: float(val) for name, val in zip(feature_names, clip_features)}
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../dataset/images/test")
    parser.add_argument("--yolo_model", default="models/best.pt")
    parser.add_argument("--output_dir", default="roi_visualizations")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of sample images")
    args = parser.parse_args()

    print("="*70)
    print("ROI Extraction Visualization")
    print("="*70)

    # 초기화
    print("\n[1] Initializing...")
    yolo_detector = YOLODetector(args.yolo_model)
    clip_encoder = CLIPEnvironmentEncoder()

    # 이미지 목록
    image_dir = Path(args.image_dir)
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return

    # 샘플 선택 (첫 N개)
    sample_images = image_files[:args.num_samples]

    print(f"\n[2] Processing {len(sample_images)} sample images...")

    results = []
    for img_path in sample_images:
        try:
            result = visualize_roi_extraction(
                img_path, yolo_detector, clip_encoder, args.output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # 요약
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    for r in results:
        print(f"\n{r['image']}:")
        print(f"  ROI size: {r['roi_size'][0]}x{r['roi_size'][1]}")
        print(f"  Baseline brightness: {r['baseline']['brightness']:.4f}")
        print(f"  Baseline contrast: {r['baseline']['contrast']:.4f}")
        print(f"  CLIP shadow: {r['clip']['clip_shadow']:.4f}")
        print(f"  CLIP clear: {r['clip']['clip_clear']:.4f}")

    print(f"\n✅ Visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
