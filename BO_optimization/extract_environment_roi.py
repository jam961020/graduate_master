"""
ROI 기반 통합 환경 추출
- Baseline features (brightness, contrast, etc.) - ROI 기반
- CLIP features (semantic) - ROI 기반
- 하나의 JSON으로 통합
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from environment_independent import extract_parameter_independent_environment
from clip_environment import CLIPEnvironmentEncoder
from yolo_detector import YOLODetector


def extract_roi_environment_all(image, yolo_detector, clip_encoder=None):
    """
    ROI 기반으로 baseline + CLIP 환경 특징 모두 추출

    Args:
        image: BGR 이미지
        yolo_detector: YOLO 검출기
        clip_encoder: CLIP 인코더 (None이면 baseline만)

    Returns:
        env_dict: {
            'brightness': float,
            'contrast': float,
            'edge_density': float,
            'texture_complexity': float,
            'blur_level': float,
            'noise_level': float,
            'clip_clear': float,  # CLIP 사용 시
            'clip_shadow': float,
            ...
        }
    """
    # 1. YOLO ROI 검출
    try:
        rois = yolo_detector.detect_rois(image)

        if len(rois) == 0:
            # No ROI, use full image
            roi_bbox = None
            roi_crop = image
        else:
            # Prefer longi_WL (class 2)
            longi_roi = [roi for roi in rois if roi[0] == 2]

            if longi_roi:
                _, x1, y1, x2, y2 = longi_roi[0]
            else:
                _, x1, y1, x2, y2 = rois[0]

            roi_bbox = (x1, y1, x2, y2)
            roi_crop = image[y1:y2, x1:x2]

            if roi_crop.size == 0:
                roi_bbox = None
                roi_crop = image

    except Exception as e:
        print(f"  [WARN] YOLO failed: {e}")
        roi_bbox = None
        roi_crop = image

    # 2. Baseline features 추출
    baseline_env = extract_parameter_independent_environment(image, roi=roi_bbox)

    # 3. CLIP features 추출 (옵션)
    env_dict = baseline_env.copy()

    if clip_encoder is not None:
        try:
            clip_features = clip_encoder.encode_roi(roi_crop)
            feature_names = clip_encoder.get_feature_names()

            for name, value in zip(feature_names, clip_features):
                env_dict[name] = float(value)

        except Exception as e:
            print(f"  [WARN] CLIP encoding failed: {e}")
            # CLIP 실패 시 baseline만 사용

    return env_dict


def batch_extract_roi_environments(image_dir, gt_file, yolo_model_path,
                                   output_file, use_clip=True):
    """
    데이터셋 전체에 대해 ROI 기반 환경 추출

    Args:
        image_dir: 이미지 디렉토리
        gt_file: Ground truth JSON (이미지 목록 가져오기용)
        yolo_model_path: YOLO 모델 경로
        output_file: 출력 JSON 파일
        use_clip: True이면 CLIP 포함, False면 baseline만
    """
    print("="*70)
    print("ROI-based Environment Feature Extraction")
    print("="*70)

    # 1. YOLO 검출기 초기화
    print("\n[1] Initializing YOLO detector...")
    yolo_detector = YOLODetector(yolo_model_path)

    # 2. CLIP 인코더 초기화 (옵션)
    clip_encoder = None
    if use_clip:
        print("\n[2] Initializing CLIP encoder...")
        clip_encoder = CLIPEnvironmentEncoder()
        print("  ✓ CLIP enabled")
    else:
        print("\n[2] CLIP disabled (baseline only)")

    # 3. Ground truth 로드
    print("\n[3] Loading ground truth...")
    with open(gt_file) as f:
        gt_data = json.load(f)

    image_names = list(gt_data.keys())
    print(f"  Found {len(image_names)} images")

    # 4. 환경 추출
    print(f"\n[4] Extracting {'baseline + CLIP' if use_clip else 'baseline'} features from ROI...")

    image_dir = Path(image_dir)
    results = {}
    roi_success = 0
    full_image_fallback = 0

    for img_name in tqdm(image_names, desc="Processing"):
        # Find image
        img_path = image_dir / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = image_dir / f"{img_name}.png"

        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_name}")
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Failed to load: {img_name}")
            continue

        # Extract environment
        try:
            env_dict = extract_roi_environment_all(image, yolo_detector, clip_encoder)
            results[img_name] = env_dict

            # Track ROI usage
            rois = yolo_detector.detect_rois(image)
            if len(rois) > 0:
                roi_success += 1
            else:
                full_image_fallback += 1

        except Exception as e:
            print(f"  [ERROR] Failed for {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 5. 결과 저장
    print(f"\n[5] Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! Extracted features for {len(results)}/{len(image_names)} images")
    print(f"  ROI detected: {roi_success}")
    print(f"  Full image fallback: {full_image_fallback}")

    # 6. 통계
    print(f"\n[6] Feature Statistics:")

    if results:
        # 모든 feature 키 가져오기
        sample_features = list(results.values())[0]
        feature_names = list(sample_features.keys())

        print(f"\n  Total features: {len(feature_names)}")

        # Baseline features
        baseline_features = ['brightness', 'contrast', 'edge_density',
                            'texture_complexity', 'blur_level', 'noise_level']

        print("\n  Baseline Features (ROI-based):")
        for fname in baseline_features:
            if fname in feature_names:
                values = [results[img][fname] for img in results if fname in results[img]]
                if values:
                    print(f"    {fname:<20}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

        # CLIP features
        if use_clip:
            clip_features = [f for f in feature_names if f.startswith('clip_')]
            if clip_features:
                print("\n  CLIP Features (ROI-based):")
                for fname in clip_features:
                    values = [results[img][fname] for img in results if fname in results[img]]
                    if values:
                        print(f"    {fname:<20}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ROI-based environment extraction")
    parser.add_argument("--image_dir", default="../dataset/images/test",
                       help="Image directory")
    parser.add_argument("--gt_file", default="../dataset/ground_truth.json",
                       help="Ground truth JSON")
    parser.add_argument("--yolo_model", default="models/best.pt",
                       help="YOLO model path")
    parser.add_argument("--output", default="environment_roi_all.json",
                       help="Output JSON file")
    parser.add_argument("--baseline_only", action="store_true",
                       help="Extract baseline features only (no CLIP)")
    args = parser.parse_args()

    use_clip = not args.baseline_only

    results = batch_extract_roi_environments(
        args.image_dir,
        args.gt_file,
        args.yolo_model,
        args.output,
        use_clip=use_clip
    )


if __name__ == "__main__":
    main()
