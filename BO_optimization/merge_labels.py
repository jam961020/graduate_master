"""
기존 ground_truth.json과 자동 라벨링 결과를 병합하는 스크립트
- 기존 GT가 있으면 우선
- 없는 이미지만 자동 라벨링 결과로 채움
"""
import json
from pathlib import Path
import argparse


def merge_ground_truths(original_gt_path, auto_gt_path, output_path, overwrite=False):
    """
    두 개의 ground truth JSON을 병합

    Args:
        original_gt_path: 기존 수동 라벨링 GT
        auto_gt_path: 자동 라벨링 GT
        output_path: 출력 파일
        overwrite: True면 자동 라벨링으로 덮어쓰기, False면 기존 우선
    """
    # 기존 GT 로드
    original_gt = {}
    if Path(original_gt_path).exists():
        with open(original_gt_path, 'r', encoding='utf-8') as f:
            original_gt = json.load(f)
        print(f"기존 GT 로드: {len(original_gt)} 이미지")
    else:
        print("기존 GT 파일 없음. 자동 라벨링만 사용.")

    # 자동 GT 로드
    auto_gt = {}
    if Path(auto_gt_path).exists():
        with open(auto_gt_path, 'r', encoding='utf-8') as f:
            auto_gt = json.load(f)
        print(f"자동 GT 로드: {len(auto_gt)} 이미지")
    else:
        print(f"자동 GT 파일 없음: {auto_gt_path}")
        return

    # 병합
    merged_gt = {}

    if overwrite:
        # 자동 라벨링 우선 (덮어쓰기 모드)
        merged_gt = {**original_gt, **auto_gt}
        print(f"[덮어쓰기 모드] 자동 라벨링으로 {len(auto_gt)} 이미지 업데이트")
    else:
        # 기존 라벨링 우선 (기본 모드)
        merged_gt = {**auto_gt, **original_gt}

        # 통계
        added_count = 0
        for img_key in auto_gt:
            if img_key not in original_gt:
                added_count += 1

        print(f"[기존 우선 모드] 새로 추가된 이미지: {added_count}")
        print(f"기존 유지: {len(original_gt)}")

    # 결과 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_gt, f, indent=2, ensure_ascii=False)

    print(f"\n병합 완료!")
    print(f"총 이미지: {len(merged_gt)}")
    print(f"저장 위치: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ground Truth 병합 스크립트")
    parser.add_argument("--original", type=str, default="../dataset/ground_truth.json",
                       help="기존 수동 라벨링 GT")
    parser.add_argument("--auto", type=str, default="../dataset/ground_truth_auto.json",
                       help="자동 라벨링 GT")
    parser.add_argument("--output", type=str, default="../dataset/ground_truth_merged.json",
                       help="병합 결과 출력")
    parser.add_argument("--overwrite", action="store_true",
                       help="자동 라벨링으로 기존 라벨 덮어쓰기")
    args = parser.parse_args()

    merge_ground_truths(args.original, args.auto, args.output, args.overwrite)


if __name__ == "__main__":
    main()
