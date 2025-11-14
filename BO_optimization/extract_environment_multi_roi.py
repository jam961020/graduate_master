"""
여러 ROI 통합 전략 실험
- first_only: 첫 번째 ROI만 (baseline)
- average: 모든 ROI 평균
- worst_case: 각 특징별 최악값 (BoRisk 철학)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from environment_independent import extract_parameter_independent_environment
from clip_environment import CLIPEnvironmentEncoder
from yolo_detector import YOLODetector


def extract_multi_roi_strategy(image, yolo_detector, clip_encoder, strategy='average'):
    """
    여러 ROI를 다양한 전략으로 종합

    Args:
        image: BGR 이미지
        yolo_detector: YOLO 검출기
        clip_encoder: CLIP 인코더
        strategy: 'first_only', 'average', 'worst_case'

    Returns:
        env_final: 13D 환경 벡터 dict
    """
    # 1. 모든 ROI 검출
    rois = yolo_detector.detect_rois(image)

    # 2. ROI 타입별 분류
    longi_rois = [roi for roi in rois if roi[0] == 2]  # class 2: longi_WL
    fillet_rois = [roi for roi in rois if roi[0] == 1]  # class 1: fillet_WL
    # collar_rois = [roi for roi in rois if roi[0] in [3, 4, 5, 6]]  # class 3-6

    # 3. 각 ROI에서 환경 추출
    env_vectors = []

    # Longi ROIs (left, right 등)
    for roi in longi_rois:
        _, x1, y1, x2, y2 = roi
        roi_bbox = (x1, y1, x2, y2)
        roi_crop = image[y1:y2, x1:x2]

        if roi_crop.size == 0:
            continue

        # Baseline 9D
        baseline = extract_parameter_independent_environment(image, roi=roi_bbox)

        # CLIP 4D
        clip = clip_encoder.encode_roi(roi_crop)

        # 통합 13D
        env = {**baseline}
        for name, val in zip(clip_encoder.get_feature_names(), clip):
            env[name] = float(val)

        env_vectors.append(env)

    # Fillet ROIs (용접 비드 영역)
    for roi in fillet_rois:
        _, x1, y1, x2, y2 = roi
        roi_bbox = (x1, y1, x2, y2)
        roi_crop = image[y1:y2, x1:x2]

        if roi_crop.size == 0:
            continue

        # Baseline 9D
        baseline = extract_parameter_independent_environment(image, roi=roi_bbox)

        # CLIP 4D
        clip = clip_encoder.encode_roi(roi_crop)

        # 통합 13D
        env = {**baseline}
        for name, val in zip(clip_encoder.get_feature_names(), clip):
            env[name] = float(val)

        env_vectors.append(env)

    # 4. 전략에 따라 종합
    if len(env_vectors) == 0:
        # No ROI detected → use full image as fallback
        print("  [WARN] No ROI detected, using full image")
        baseline = extract_parameter_independent_environment(image, roi=None)
        clip = clip_encoder.encode_roi(image)
        env_final = {**baseline}
        for name, val in zip(clip_encoder.get_feature_names(), clip):
            env_final[name] = float(val)
        return env_final

    if strategy == 'first_only':
        # 첫 번째 ROI만 (baseline)
        return env_vectors[0]

    elif strategy == 'average':
        # 평균
        env_final = {}
        feature_names = list(env_vectors[0].keys())

        for fname in feature_names:
            values = [env[fname] for env in env_vectors]
            env_final[fname] = float(np.mean(values))

        return env_final

    elif strategy == 'worst_case':
        # 각 특징별 worst (높은 값 = 어려움)
        env_final = {}
        feature_names = list(env_vectors[0].keys())

        for fname in feature_names:
            values = [env[fname] for env in env_vectors]
            env_final[fname] = float(np.max(values))  # worst = max

        return env_final

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    strategies = ['first_only', 'average', 'worst_case']

    print("="*70)
    print("Multi-ROI Strategy Experiment")
    print("="*70)

    # 1. 초기화
    print("\n[1] Initializing models...")
    yolo_detector = YOLODetector("models/best.pt")
    clip_encoder = CLIPEnvironmentEncoder()

    # 2. GT 데이터 로드
    print("\n[2] Loading ground truth...")
    with open("../dataset/ground_truth.json") as f:
        gt_data = json.load(f)
    image_names = list(gt_data.keys())
    print(f"  Found {len(image_names)} images")

    image_dir = Path("../dataset/images/test")

    # 3. 각 전략별로 환경 추출
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy}")
        print(f"{'='*70}")

        results = {}
        num_multi_roi = 0  # 여러 ROI 사용한 이미지 개수

        for img_name in tqdm(image_names, desc=strategy):
            # Find image
            img_path = image_dir / f"{img_name}.jpg"
            if not img_path.exists():
                img_path = image_dir / f"{img_name}.png"
            if not img_path.exists():
                continue

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            try:
                # 환경 추출
                env = extract_multi_roi_strategy(image, yolo_detector, clip_encoder, strategy)
                results[img_name] = env

                # ROI 개수 확인 (통계용)
                rois = yolo_detector.detect_rois(image)
                longi_count = len([r for r in rois if r[0] == 2])
                fillet_count = len([r for r in rois if r[0] == 1])
                total_count = longi_count + fillet_count

                if total_count > 1:
                    num_multi_roi += 1

            except Exception as e:
                print(f"  [ERROR] {img_name}: {e}")
                import traceback
                traceback.print_exc()

        # 4. 저장
        output_file = f"environment_roi_{strategy}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  ✅ Saved to {output_file}")
        print(f"  Processed: {len(results)}/{len(image_names)} images")
        print(f"  Multi-ROI images: {num_multi_roi}")

        # 5. 간단한 통계
        if results:
            sample_env = list(results.values())[0]
            feature_names = list(sample_env.keys())

            print(f"\n  Feature Statistics:")
            for fname in feature_names[:5]:  # 처음 5개만 출력
                values = [results[img][fname] for img in results if fname in results[img]]
                if values:
                    print(f"    {fname:<25}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    print("\n" + "="*70)
    print("✅ All strategies completed!")
    print("="*70)
    print("\nNext step:")
    print("  python analyze_clip_correlation.py --clip_features environment_roi_first_only.json")
    print("  python analyze_clip_correlation.py --clip_features environment_roi_average.json")
    print("  python analyze_clip_correlation.py --clip_features environment_roi_worst_case.json")


if __name__ == "__main__":
    main()
