"""
디버그용 검출 결과 시각화 및 저장
실험 중 score 0 케이스 분석용
"""

import cv2
import numpy as np
from pathlib import Path
import json


def save_detection_debug(image, detected_coords, gt_coords, image_name,
                         iteration, score, save_dir, yolo_rois=None):
    """
    검출 결과를 시각화하여 저장

    Args:
        image: BGR 이미지
        detected_coords: 검출된 12개 좌표 dict
        gt_coords: GT 12개 좌표 dict
        image_name: 이미지 이름
        iteration: 현재 iteration
        score: 평가 점수
        save_dir: 저장 디렉토리
        yolo_rois: YOLO bbox 리스트 (optional)
    """
    debug_dir = Path(save_dir) / "debug_images"
    debug_dir.mkdir(exist_ok=True)

    # 이미지 복사
    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    # 1. YOLO ROI 그리기 (있으면)
    if yolo_rois:
        for cls, x1, y1, x2, y2 in yolo_rois:
            color = (255, 255, 0)  # Cyan
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"cls{cls}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2. GT 선 그리기 (초록색)
    if gt_coords:
        gt_lines = extract_lines_from_coords(gt_coords)
        for (x1, y1), (x2, y2) in gt_lines:
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 255, 0), 3)  # Green, thick
        # GT 점 표시
        for (x1, y1), (x2, y2) in gt_lines:
            cv2.circle(vis_img, (int(x1), int(y1)), 5, (0, 255, 0), -1)
            cv2.circle(vis_img, (int(x2), int(y2)), 5, (0, 255, 0), -1)

    # 3. 검출된 선 그리기 (빨간색)
    if detected_coords:
        det_lines = extract_lines_from_coords(detected_coords)
        for (x1, y1), (x2, y2) in det_lines:
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 0, 255), 2)  # Red
        # 검출 점 표시
        for (x1, y1), (x2, y2) in det_lines:
            cv2.circle(vis_img, (int(x1), int(y1)), 4, (0, 0, 255), -1)
            cv2.circle(vis_img, (int(x2), int(y2)), 4, (0, 0, 255), -1)

    # 4. 정보 텍스트
    info_text = f"Iter {iteration} | Score: {score:.4f}"
    cv2.putText(vis_img, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_img, image_name, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 범례
    cv2.putText(vis_img, "Green: GT", (10, h-40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_img, "Red: Detected", (10, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 5. 저장 (모든 iteration 저장, score 표시)
    score_str = f"{score:.3f}".replace('.', 'p')
    if score == 0:
        filename = f"iter_{iteration:03d}_FAIL_s{score_str}_{image_name}.jpg"
    else:
        filename = f"iter_{iteration:03d}_s{score_str}_{image_name}.jpg"

    save_path = debug_dir / filename
    cv2.imwrite(str(save_path), vis_img)

    return str(save_path)


def get_yolo_rois(image, yolo_detector, conf_threshold=0.6):
    """YOLO ROI 검출 결과 반환"""
    rois = yolo_detector.detect_rois(image, conf_threshold=conf_threshold)
    return rois


def extract_lines_from_coords(coords):
    """좌표에서 선 추출 (최대 5개 선)

    점 6개 구조:
    longi_left_upper    collar_left_upper    longi_right_upper
           |                   |                   |
    longi_left_lower -- collar_left_lower -- longi_right_lower

    선 5개:
    1. longi_left 세로선
    2. longi_left_lower → collar_left_lower 가로선
    3. collar_left_lower → longi_right_lower 가로선
    4. collar_left 세로선
    5. longi_right 세로선
    """
    lines = []

    # 점 좌표 추출 (없으면 None)
    def get_point(prefix):
        x = coords.get(f"{prefix}_x", 0)
        y = coords.get(f"{prefix}_y", 0)
        if x == 0 and y == 0:
            return None
        return (x, y)

    longi_left_lower = get_point("longi_left_lower")
    longi_left_upper = get_point("longi_left_upper")
    collar_left_lower = get_point("collar_left_lower")
    collar_left_upper = get_point("collar_left_upper")
    longi_right_lower = get_point("longi_right_lower")
    longi_right_upper = get_point("longi_right_upper")

    # 1. longi_left 세로선
    if longi_left_lower and longi_left_upper:
        lines.append((longi_left_lower, longi_left_upper))

    # 2. 가로선 왼쪽 (longi_left_lower → collar_left_lower)
    if longi_left_lower and collar_left_lower:
        lines.append((longi_left_lower, collar_left_lower))

    # 3. 가로선 오른쪽 (collar_left_lower → longi_right_lower)
    if collar_left_lower and longi_right_lower:
        lines.append((collar_left_lower, longi_right_lower))

    # 4. collar_left 세로선
    if collar_left_lower and collar_left_upper:
        lines.append((collar_left_lower, collar_left_upper))

    # 5. longi_right 세로선
    if longi_right_lower and longi_right_upper:
        lines.append((longi_right_lower, longi_right_upper))

    return lines


def check_detection_failure(detected_coords):
    """검출 실패 여부 확인"""
    if detected_coords is None:
        return True, "No coords returned"

    lines = extract_lines_from_coords(detected_coords)
    if len(lines) == 0:
        return True, "No lines detected"

    # 모든 좌표가 0인지 확인
    all_zero = all(
        detected_coords.get(k, 0) == 0
        for k in detected_coords.keys()
    )
    if all_zero:
        return True, "All coords are zero"

    return False, "OK"


# optimization.py의 evaluate_single에 추가할 코드 예시
INTEGRATION_CODE = '''
# === optimization.py에 추가할 코드 ===

# 1. import 추가 (파일 상단, line ~30)
from debug_visualizer import save_detection_debug, get_yolo_rois

# 2. evaluate_single 함수 수정 (line 387~)
def evaluate_single(X, image_data, yolo_detector, iteration=None, save_dir=None):
    """
    단일 (x, w) 쌍만 평가 (진짜 BoRisk!)

    추가 Args:
        iteration: 현재 iteration (디버그용)
        save_dir: 로그 저장 디렉토리 (디버그용)
    """
    params = {
        'edgeThresh1': X[0, 0].item(),
        'simThresh1': X[0, 1].item(),
        'pixelRatio1': X[0, 2].item(),
        'edgeThresh2': X[0, 3].item(),
        'simThresh2': X[0, 4].item(),
        'pixelRatio2': X[0, 5].item(),
    }
    ransac_weights = (X[0, 6].item(), X[0, 7].item())

    image = image_data['image']
    gt_coords = image_data['gt_coords']

    # YOLO ROI 먼저 획득 (디버그용)
    yolo_rois = get_yolo_rois(image, yolo_detector) if save_dir else None

    detected_coords = detect_with_full_pipeline(image, params, yolo_detector, ransac_weights)

    # Metric 계산
    from evaluation import evaluate_lp
    score = evaluate_lp(detected_coords, image, image_data.get('name'), threshold=30.0, debug=False)

    # 디버그 시각화 저장 (모든 iteration)
    if save_dir and iteration is not None:
        save_detection_debug(
            image=image,
            detected_coords=detected_coords,
            gt_coords=gt_coords,
            image_name=image_data.get('name', 'unknown'),
            iteration=iteration,
            score=score,
            save_dir=save_dir,
            yolo_rois=yolo_rois
        )

    return score

# 3. BO 루프에서 호출 시 (line ~870)
# score = evaluate_single(candidate, images_data[img_idx], yolo_detector)
# 를 다음으로 변경:
score = evaluate_single(
    candidate,
    images_data[img_idx],
    yolo_detector,
    iteration=i,  # 현재 iteration
    save_dir=log_dir  # 로그 디렉토리
)
'''


if __name__ == "__main__":
    print("Debug Visualizer for BO Optimization")
    print("=" * 50)
    print("\n통합 방법:")
    print(INTEGRATION_CODE)
