"""
Score=0이 발생하는 문제 이미지 찾기
기본 파라미터로 모든 이미지를 평가하여 문제 이미지 식별
"""
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

from full_pipeline import detect_lines_in_roi, weighted_ransac_line
from yolo_detector import YOLODetector
from evaluation import evaluate_lp
import cv2

def evaluate_single_image(image, gt_coords, yolo_detector, params):
    """단일 이미지 평가"""
    try:
        # YOLO로 ROI 검출
        rois = yolo_detector.detect_rois(image)
        if rois is None or len(rois) == 0:
            return 0.0, "YOLO failed"

        # detect_rois는 (class_id, x1, y1, x2, y2) 반환
        _, x1, y1, x2, y2 = rois[0]  # 첫 번째 ROI 사용
        roi_image = image[y1:y2, x1:x2]

        # AirLine으로 라인 검출
        lines_Q, lines_QG = detect_lines_in_roi(
            roi_image,
            edgeThresh1=params['edgeThresh1'],
            simThresh1=params['simThresh1'],
            pixelRatio1=params['pixelRatio1'],
            edgeThresh2=params['edgeThresh2'],
            simThresh2=params['simThresh2'],
            pixelRatio2=params['pixelRatio2']
        )

        if lines_Q is None or lines_QG is None:
            return 0.0, "AirLine failed"

        if len(lines_Q) < 2 or len(lines_QG) < 2:
            return 0.0, f"Not enough lines: Q={len(lines_Q)}, QG={len(lines_QG)}"

        # RANSAC
        all_lines = lines_Q + lines_QG
        weights = [params['ransac_weight_q']] * len(lines_Q) + \
                  [params['ransac_weight_qg']] * len(lines_QG)

        final_lines = weighted_ransac_line(all_lines, weights)

        if final_lines is None or len(final_lines) < 4:
            return 0.0, f"RANSAC failed: {len(final_lines) if final_lines else 0} lines"

        # 좌표 변환
        detected_coords = {}
        for i, line in enumerate(final_lines[:6]):
            x, y = line[0], line[1]
            detected_coords[f'p{i+1}'] = [int(x + x1), int(y + y1)]

        # 평가
        score = evaluate_lp(detected_coords, image, threshold=20.0)
        return score, "OK"

    except Exception as e:
        return 0.0, str(e)


def main():
    # 기본 파라미터 (여러 세트 테스트)
    param_sets = [
        # 기본값
        {
            'edgeThresh1': -3.0, 'simThresh1': 0.98, 'pixelRatio1': 0.05,
            'edgeThresh2': 1.0, 'simThresh2': 0.75, 'pixelRatio2': 0.05,
            'ransac_weight_q': 5.0, 'ransac_weight_qg': 5.0
        },
        # 더 관대한 설정
        {
            'edgeThresh1': -10.0, 'simThresh1': 0.7, 'pixelRatio1': 0.1,
            'edgeThresh2': -5.0, 'simThresh2': 0.6, 'pixelRatio2': 0.1,
            'ransac_weight_q': 10.0, 'ransac_weight_qg': 10.0
        },
        # 더 엄격한 설정
        {
            'edgeThresh1': 0.0, 'simThresh1': 0.9, 'pixelRatio1': 0.03,
            'edgeThresh2': 3.0, 'simThresh2': 0.85, 'pixelRatio2': 0.03,
            'ransac_weight_q': 3.0, 'ransac_weight_qg': 3.0
        }
    ]

    # 데이터 로드
    image_dir = Path('../dataset/images/for_BO')
    gt_file = Path('../dataset/ground_truth_merged.json')

    with open(gt_file) as f:
        gt_data = json.load(f)

    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    print(f"Total images: {len(image_files)}")

    # YOLO 로드
    yolo_detector = YOLODetector('models/best.pt')

    # 결과 저장
    bad_images = []  # 모든 파라미터에서 실패
    partial_bad = []  # 일부 파라미터에서 실패

    print("\nEvaluating images...")
    for idx, img_path in enumerate(tqdm(image_files[:600])):
        img_name = img_path.stem

        # GT 확인
        if img_name not in gt_data:
            continue

        gt_coords = gt_data[img_name]
        image = cv2.imread(str(img_path))

        if image is None:
            bad_images.append((idx, img_name, "Cannot read image"))
            continue

        # 모든 파라미터 세트로 테스트
        scores = []
        failures = []

        for i, params in enumerate(param_sets):
            score, msg = evaluate_single_image(image, gt_coords, yolo_detector, params)
            scores.append(score)
            if score == 0:
                failures.append((i, msg))

        # 결과 분류
        if all(s == 0 for s in scores):
            bad_images.append((idx, img_name, failures[0][1]))
        elif any(s == 0 for s in scores):
            partial_bad.append((idx, img_name, len(failures), scores))

    # 결과 출력
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\n## Bad images (fail in ALL param sets): {len(bad_images)}")
    for idx, name, reason in bad_images:
        print(f"  idx={idx}: {name} - {reason}")

    print(f"\n## Partial bad images (fail in SOME param sets): {len(partial_bad)}")
    for idx, name, n_fail, scores in partial_bad[:20]:  # 처음 20개만
        print(f"  idx={idx}: {name} - {n_fail}/3 failed, scores={scores}")

    # JSON으로 저장
    result = {
        'bad_images': [(idx, name) for idx, name, _ in bad_images],
        'bad_image_indices': [idx for idx, _, _ in bad_images],
        'partial_bad_indices': [idx for idx, _, _, _ in partial_bad]
    }

    with open('bad_images.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n## Saved to bad_images.json")
    print(f"## Exclude these {len(bad_images)} images and re-run experiment")


if __name__ == "__main__":
    main()
