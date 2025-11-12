"""
자동 라벨링 스크립트
full_pipeline.py의 detect_with_full_pipeline을 사용하여
6개 점(longi 4개 + collar 2개)을 자동 추출
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# full_pipeline import
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector


def auto_label_image(image_path, yolo_detector, params):
    """
    이미지를 자동으로 라벨링 (detect_with_full_pipeline 사용)

    Args:
        image_path: 이미지 경로
        yolo_detector: YOLO 검출기
        params: AirLine 파라미터 dict

    Returns:
        coords: dict with 12 coordinates (6 points * 2)
        or None if labeling failed
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return None

    # detect_with_full_pipeline 실행
    try:
        coords = detect_with_full_pipeline(image, params, yolo_detector)

        # 유효성 검사: 최소 4개 점이 검출되었는지
        non_zero_count = sum(1 for k, v in coords.items() if v != 0 and 'x' in k)
        if non_zero_count < 4:
            print(f"[WARN] {image_path.name}: Only {non_zero_count}/6 points detected")
            return None

        return coords

    except Exception as e:
        print(f"[ERROR] {image_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="자동 라벨링 스크립트")
    parser.add_argument("--image_dir", type=str, default="../dataset/images/test",
                       help="이미지 디렉토리")
    parser.add_argument("--output", type=str, default="../dataset/ground_truth_auto.json",
                       help="출력 JSON 파일")
    parser.add_argument("--yolo_model", type=str, default="models/best.pt",
                       help="YOLO 모델 경로")
    parser.add_argument("--max_images", type=int, default=None,
                       help="최대 처리 이미지 수 (테스트용)")
    args = parser.parse_args()

    # YOLO 검출기 초기화
    print("YOLO 검출기 초기화 중...")
    yolo_detector = YOLODetector(args.yolo_model)

    # 기본 파라미터 (중간값)
    params = {
        'edgeThresh1': -8.0,
        'simThresh1': 0.75,
        'pixelRatio1': 0.08,
        'edgeThresh2': -8.0,
        'simThresh2': 0.75,
        'pixelRatio2': 0.08,
    }

    # 이미지 로드
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if args.max_images:
        image_files = image_files[:args.max_images]
    print(f"Found {len(image_files)} images")

    # 자동 라벨링 실행
    # ground_truth.json과 동일한 포맷: {이미지명(확장자 제외): {좌표}}
    results = {}
    success_count = 0

    for image_path in tqdm(image_files, desc="Auto-labeling"):
        coords = auto_label_image(image_path, yolo_detector, params)
        if coords is not None:
            # 키는 확장자 제외 (ground_truth.json과 동일)
            img_key = image_path.stem
            results[img_key] = coords
            success_count += 1
        else:
            print(f"  Failed: {image_path.name}")

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n자동 라벨링 완료!")
    print(f"성공: {success_count}/{len(image_files)} ({success_count/len(image_files)*100:.1f}%)")
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
