"""
실질적 난이도 기반 환경 변수 추출
"""
import cv2
import numpy as np
import sys
import torch
from pathlib import Path

# 기존 코드 import - 경로만 추가하고 직접 import
YOLO_PATH = Path(__file__).parent.parent / "YOLO_AirLine"
sys.path.insert(0, str(YOLO_PATH))

# 파일명에 따라 import 시도
try:
    # 방법 1: test_clone.py인 경우
    from test_clone import EDGE_DET, ORI_DET, THETA_RES
    print("[INFO] Imported from test_clone.py")
except ImportError:
    try:
        # 방법 2: AirLine_assemble_test.py인 경우
        from AirLine_assemble_test import EDGE_DET, ORI_DET, THETA_RES
        print("[INFO] Imported from AirLine_assemble_test.py")
    except ImportError as e:
        print(f"[ERROR] Failed to import EDGE_DET, ORI_DET, THETA_RES: {e}")
        print(f"[ERROR] Path checked: {YOLO_PATH}")
        print(f"[ERROR] Files in directory:")
        if YOLO_PATH.exists():
            for f in YOLO_PATH.glob("*.py"):
                print(f"  - {f.name}")
        else:
            print(f"  Directory not found: {YOLO_PATH}")
        raise


def compute_continuity_break(image, roi, airline_config):
    """
    DexiNed 확률 맵에서 선의 끊김 정도 측정
    
    Args:
        image: BGR 이미지
        roi: (x1, y1, x2, y2) 또는 None (전체)
        airline_config: edge detection용
    
    Returns:
        break_score: 0~1 (높을수록 많이 끊김, 어려운 환경)
    """
    # ROI 추출
    if roi:
        x1, y1, x2, y2 = roi
        roi_img = image[y1:y2, x1:x2]
    else:
        roi_img = image
    
    h, w = roi_img.shape[:2]
    
    # DexiNed로 확률 맵 생성
    res = 16
    dscale = 1
    rh = h // dscale // res * res
    rw = w // dscale // res * res
    
    if rh == 0 or rw == 0:
        return 0.5  # 기본값
    
    resized = cv2.resize(roi_img, (rw, rh))
    resized = np.ascontiguousarray(resized)
    x = torch.tensor(resized, dtype=torch.float32).cuda() / 255.0
    x = x.permute(2, 0, 1)
    
    with torch.no_grad():
        edge_map = EDGE_DET(x.unsqueeze(0))
    
    edge_np = edge_map.detach().cpu().numpy()[0, 0]
    
    # Threshold 적용하여 이진화
    threshold = airline_config.get('edgeThresh', 0)
    edge_binary = (edge_np > threshold).astype(np.uint8) * 255
    
    # 연결 요소 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_binary, connectivity=8
    )
    
    if num_labels <= 1:  # 배경만 있음
        return 1.0  # 완전히 끊김
    
    # 가장 큰 연결 요소 크기
    areas = stats[1:, cv2.CC_STAT_AREA]  # 배경 제외
    max_area = areas.max() if len(areas) > 0 else 0
    total_area = edge_binary.sum() / 255
    
    # 파편화 정도
    if total_area == 0:
        return 1.0
    
    continuity_ratio = max_area / total_area
    
    # 반전: 1 - ratio = 끊김 정도
    break_score = 1.0 - continuity_ratio
    
    return float(np.clip(break_score, 0.0, 1.0))


def compute_orientation_variance(image, roi, airline_config):
    """
    Orient 맵에서 방향 분산 계산
    
    Args:
        image: BGR 이미지
        roi: (x1, y1, x2, y2) 또는 None
        airline_config: 설정
    
    Returns:
        variance_score: 0~1 (높을수록 방향 불규칙, 어려움)
    """
    # ROI 추출
    if roi:
        x1, y1, x2, y2 = roi
        roi_img = image[y1:y2, x1:x2]
    else:
        roi_img = image
    
    h, w = roi_img.shape[:2]
    
    # 전처리
    res = 16
    dscale = 1
    rh = h // dscale // res * res
    rw = w // dscale // res * res
    
    if rh == 0 or rw == 0:
        return 0.5
    
    resized = cv2.resize(roi_img, (rw, rh))
    resized = np.ascontiguousarray(resized)
    x = torch.tensor(resized, dtype=torch.float32).cuda() / 255.0
    x = x.permute(2, 0, 1)
    
    with torch.no_grad():
        edge_map = EDGE_DET(x.unsqueeze(0))
        orient_map = ORI_DET(edge_map)
    
    # Normalize orient
    orient_map = torch.nn.functional.normalize(
        orient_map - orient_map.mean(1, keepdim=True), 
        p=2.0, 
        dim=1
    )
    
    orient_np = orient_map[0].detach().cpu().numpy()  # [THETA_RES, H, W]
    
    # Edge binary mask
    edge_np = edge_map.detach().cpu().numpy()[0, 0]
    threshold = airline_config.get('edgeThresh', 0)
    edge_mask = edge_np > threshold
    
    if edge_mask.sum() == 0:
        return 0.5
    
    # 각 픽셀의 dominant orientation
    dominant_orient = orient_np.argmax(axis=0)  # [H, W]
    
    # Edge 픽셀의 orientation만 추출
    edge_orients = dominant_orient[edge_mask]
    
    # Circular variance (각도는 circular data)
    angles_deg = edge_orients * (180.0 / THETA_RES)
    angles_rad = np.deg2rad(angles_deg)
    
    # Circular mean
    sin_mean = np.sin(angles_rad).mean()
    cos_mean = np.cos(angles_rad).mean()
    
    # Circular variance: 1 - R
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    circular_var = 1.0 - R
    
    # 정규화
    variance_score = float(np.clip(circular_var, 0.0, 1.0))
    
    return variance_score


def compute_illumination_unevenness(image, roi):
    """
    조도 불균일 정도 측정 (그림자 강도)
    
    Args:
        image: BGR 이미지
        roi: (x1, y1, x2, y2) 또는 None
    
    Returns:
        unevenness_score: 0~1 (높을수록 불균일, 어려움)
    """
    # ROI 추출
    if roi:
        x1, y1, x2, y2 = roi
        roi_img = image[y1:y2, x1:x2]
    else:
        roi_img = image
    
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # 방법 1: 밝기 표준편차 / 평균
    mean_brightness = gray.mean()
    std_brightness = gray.std()
    
    if mean_brightness < 10:  # 너무 어두우면
        return 1.0
    
    cv_brightness = std_brightness / mean_brightness  # Coefficient of Variation
    
    # 정규화 (경험적으로 0.5 이상이면 매우 불균일)
    unevenness_score = np.clip(cv_brightness / 0.5, 0.0, 1.0)
    
    # 방법 2: 로컬 대비
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_diff = np.abs(gray.astype(np.float32) - local_mean)
    
    # 강한 그림자 영역 비율
    strong_shadow_ratio = (local_diff > 50).sum() / gray.size
    
    # 두 지표 결합
    combined_score = 0.6 * unevenness_score + 0.4 * strong_shadow_ratio
    
    return float(np.clip(combined_score, 0.0, 1.0))


def compute_centroid_dispersion(image, roi, airline_config):
    """
    검출된 edge 픽셀들의 공간적 분산 측정
    
    Args:
        image: BGR 이미지
        roi: (x1, y1, x2, y2) 또는 None
        airline_config: 설정
    
    Returns:
        dispersion_score: 0~1 (높을수록 분산, 어려움)
    """
    # ROI 추출
    if roi:
        x1, y1, x2, y2 = roi
        roi_img = image[y1:y2, x1:x2]
    else:
        roi_img = image
    
    h, w = roi_img.shape[:2]
    
    # Edge 검출
    res = 16
    dscale = 1
    rh = h // dscale // res * res
    rw = w // dscale // res * res
    
    if rh == 0 or rw == 0:
        return 0.5
    
    resized = cv2.resize(roi_img, (rw, rh))
    resized = np.ascontiguousarray(resized)
    x = torch.tensor(resized, dtype=torch.float32).cuda() / 255.0
    x = x.permute(2, 0, 1)
    
    with torch.no_grad():
        edge_map = EDGE_DET(x.unsqueeze(0))
    
    edge_np = edge_map.detach().cpu().numpy()[0, 0]
    threshold = airline_config.get('edgeThresh', 0)
    edge_binary = edge_np > threshold
    
    # Edge 픽셀 좌표
    edge_coords = np.argwhere(edge_binary)  # [N, 2] (y, x)
    
    if len(edge_coords) < 10:
        return 1.0  # 너무 적으면 어려움
    
    # 중심점
    centroid = edge_coords.mean(axis=0)
    
    # 각 점에서 중심까지 거리
    distances = np.linalg.norm(edge_coords - centroid, axis=1)
    
    # 거리의 표준편차
    mean_dist = distances.mean()
    std_dist = distances.std()
    
    if mean_dist < 1:
        return 0.5
    
    # Coefficient of Variation
    cv_dist = std_dist / mean_dist
    
    # 정규화 (경험적으로 1.5 이상이면 매우 분산)
    dispersion_score = np.clip(cv_dist / 1.5, 0.0, 1.0)
    
    return float(dispersion_score)


def extract_environment_advanced(image, roi=None, airline_config=None):
    """
    4가지 실질적 난이도 지표 추출
    
    Args:
        image: BGR 이미지
        roi: (x1, y1, x2, y2) 또는 None (전체)
        airline_config: AirLine 설정 (edgeThresh 등)
    
    Returns:
        env: {
            'continuity_break': float [0, 1],
            'orientation_var': float [0, 1],
            'illumination_uneven': float [0, 1],
            'centroid_dispersion': float [0, 1],
            'difficulty': float [0, 1]  # 종합 난이도
        }
    """
    if airline_config is None:
        airline_config = {'edgeThresh': 0}
    
    # 4가지 지표 계산
    try:
        continuity = compute_continuity_break(image, roi, airline_config)
    except Exception as e:
        print(f"[WARN] continuity_break failed: {e}")
        continuity = 0.5
    
    try:
        orientation = compute_orientation_variance(image, roi, airline_config)
    except Exception as e:
        print(f"[WARN] orientation_variance failed: {e}")
        orientation = 0.5
    
    try:
        illumination = compute_illumination_unevenness(image, roi)
    except Exception as e:
        print(f"[WARN] illumination_unevenness failed: {e}")
        illumination = 0.5
    
    try:
        dispersion = compute_centroid_dispersion(image, roi, airline_config)
    except Exception as e:
        print(f"[WARN] centroid_dispersion failed: {e}")
        dispersion = 0.5
    
    # 종합 난이도 (가중 평균)
    w = [0.3, 0.25, 0.25, 0.2]  # continuity, orientation, illumination, dispersion
    difficulty = (
        w[0] * continuity +
        w[1] * orientation +
        w[2] * illumination +
        w[3] * dispersion
    )
    
    return {
        'continuity_break': float(continuity),
        'orientation_var': float(orientation),
        'illumination_uneven': float(illumination),
        'centroid_dispersion': float(dispersion),
        'difficulty': float(difficulty)
    }


def batch_extract_environments(image_dir, output_file, airline_config=None):
    """
    폴더 내 모든 이미지의 환경 정보 추출
    
    Args:
        image_dir: 이미지 폴더 경로
        output_file: 출력 JSON 파일
        airline_config: AirLine 설정
    """
    import json
    from tqdm import tqdm
    
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    results = {}
    
    for img_path in tqdm(images, desc="Extracting environments"):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            env = extract_environment_advanced(image, roi=None, airline_config=airline_config)
            results[img_path.stem] = env
            
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue
    
    # 저장
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    
    # 통계
    if results:
        difficulties = [data['difficulty'] for data in results.values()]
        continuities = [data['continuity_break'] for data in results.values()]
        orientations = [data['orientation_var'] for data in results.values()]
        illuminations = [data['illumination_uneven'] for data in results.values()]
        dispersions = [data['centroid_dispersion'] for data in results.values()]
        
        print(f"\n=== 환경 통계 ({len(results)}장) ===")
        print(f"종합 난이도:     {min(difficulties):.3f} ~ {max(difficulties):.3f} (평균: {np.mean(difficulties):.3f})")
        print(f"연속성 붕괴:     {min(continuities):.3f} ~ {max(continuities):.3f} (평균: {np.mean(continuities):.3f})")
        print(f"방향 분산:       {min(orientations):.3f} ~ {max(orientations):.3f} (평균: {np.mean(orientations):.3f})")
        print(f"조도 불균일:     {min(illuminations):.3f} ~ {max(illuminations):.3f} (평균: {np.mean(illuminations):.3f})")
        print(f"중심점 분산:     {min(dispersions):.3f} ~ {max(dispersions):.3f} (평균: {np.mean(dispersions):.3f})")


# 테스트
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="환경 난이도 추출 도구")
    parser.add_argument("input", help="이미지 파일 또는 폴더")
    parser.add_argument("--edge_thresh", type=float, default=0, help="Edge threshold")
    parser.add_argument("--output", help="출력 JSON 파일 (폴더 모드)")
    parser.add_argument("--batch", action="store_true", help="폴더 전체 처리")
    args = parser.parse_args()
    
    config = {'edgeThresh': args.edge_thresh}
    
    if args.batch:
        # 배치 모드
        output_file = args.output if args.output else "environment_analysis.json"
        batch_extract_environments(args.input, output_file, config)
    
    else:
        # 단일 이미지 모드
        image = cv2.imread(args.input)
        if image is None:
            print(f"Cannot load: {args.input}")
            exit(1)
        
        env = extract_environment_advanced(image, roi=None, airline_config=config)
        
        print(f"\n{'='*50}")
        print(f"환경 난이도 분석: {Path(args.input).name}")
        print(f"{'='*50}")
        
        print(f"\n연속성 붕괴:     {env['continuity_break']:.3f}", end="")
        if env['continuity_break'] > 0.7:
            print(" ⚠️ 심각 (선이 많이 끊김)")
        elif env['continuity_break'] > 0.4:
            print(" ⚠️ 중간 (선이 약간 끊김)")
        else:
            print(" ✓ 양호 (연속적)")
        
        print(f"방향 분산:       {env['orientation_var']:.3f}", end="")
        if env['orientation_var'] > 0.7:
            print(" ⚠️ 불규칙 (방향 일정하지 않음)")
        elif env['orientation_var'] > 0.4:
            print(" ⚠️ 중간")
        else:
            print(" ✓ 일정 (직선)")
        
        print(f"조도 불균일:     {env['illumination_uneven']:.3f}", end="")
        if env['illumination_uneven'] > 0.7:
            print(" ⚠️ 강한 그림자")
        elif env['illumination_uneven'] > 0.4:
            print(" ⚠️ 중간")
        else:
            print(" ✓ 균일한 조명")
        
        print(f"중심점 분산:     {env['centroid_dispersion']:.3f}", end="")
        if env['centroid_dispersion'] > 0.7:
            print(" ⚠️ 노이즈 많음")
        elif env['centroid_dispersion'] > 0.4:
            print(" ⚠️ 중간")
        else:
            print(" ✓ 깨끗함")
        
        print(f"\n{'='*50}")
        print(f"종합 난이도:     {env['difficulty']:.3f}", end="")
        if env['difficulty'] > 0.7:
            print(" ⚠️⚠️ 매우 어려움")
        elif env['difficulty'] > 0.5:
            print(" ⚠️ 어려움")
        elif env['difficulty'] > 0.3:
            print(" △ 보통")
        else:
            print(" ✓ 쉬움")
        print(f"{'='*50}\n")