"""
라벨 검증 및 통계 스크립트
ground_truth.json의 완성도를 확인
"""
import json
from pathlib import Path
import argparse


def check_labels(gt_path, image_dir):
    """
    Ground truth 완성도 체크

    Args:
        gt_path: ground_truth.json 경로
        image_dir: 이미지 디렉토리
    """
    # GT 로드
    if not Path(gt_path).exists():
        print(f"GT 파일 없음: {gt_path}")
        return

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    print("="*60)
    print(f"Ground Truth 분석: {gt_path}")
    print("="*60)

    # 이미지 파일 리스트
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_names = {img.stem for img in image_files}

    print(f"\n전체 이미지: {len(image_files)}")
    print(f"라벨링된 이미지: {len(gt_data)}")
    print(f"라벨링 비율: {len(gt_data)/len(image_files)*100:.1f}%")

    # 라벨링 완성도 분석
    complete_count = 0  # 6개 점 모두 있음
    partial_count = 0   # 일부만 있음
    missing_images = []

    for img_name in image_names:
        if img_name not in gt_data:
            missing_images.append(img_name)
            continue

        coords = gt_data[img_name]

        # 6개 점 체크
        required_keys = [
            'longi_left_lower_x', 'longi_left_lower_y',
            'longi_left_upper_x', 'longi_left_upper_y',
            'longi_right_lower_x', 'longi_right_lower_y',
            'longi_right_upper_x', 'longi_right_upper_y',
            'collar_left_lower_x', 'collar_left_lower_y',
            'collar_left_upper_x', 'collar_left_upper_y'
        ]

        non_zero_count = sum(1 for key in required_keys if coords.get(key, 0) != 0)

        if non_zero_count == 12:
            complete_count += 1
        elif non_zero_count > 0:
            partial_count += 1

    print(f"\n완전한 라벨 (6개 점): {complete_count}")
    print(f"부분 라벨 (1-5개 점): {partial_count}")
    print(f"라벨 없음: {len(missing_images)}")

    # 라벨 없는 이미지 목록
    if missing_images:
        print(f"\n라벨 없는 이미지 ({len(missing_images)}개):")
        for i, img_name in enumerate(missing_images[:20], 1):  # 최대 20개만 출력
            print(f"  {i}. {img_name}")
        if len(missing_images) > 20:
            print(f"  ... 외 {len(missing_images) - 20}개")

    print("\n" + "="*60)
    print("권장 작업:")
    if len(missing_images) > 0:
        print(f"1. 자동 라벨링으로 {len(missing_images)}개 이미지 추가:")
        print(f"   python auto_labeling.py --output ../dataset/ground_truth_auto.json")
        print(f"2. 병합:")
        print(f"   python merge_labels.py --output ../dataset/ground_truth_merged.json")
    else:
        print("모든 이미지가 라벨링되었습니다! ✓")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Ground Truth 검증 스크립트")
    parser.add_argument("--gt_file", type=str, default="../dataset/ground_truth.json",
                       help="Ground truth JSON 파일")
    parser.add_argument("--image_dir", type=str, default="../dataset/images/test",
                       help="이미지 디렉토리")
    args = parser.parse_args()

    check_labels(args.gt_file, args.image_dir)


if __name__ == "__main__":
    main()
