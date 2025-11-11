"""
ROI 및 전체 이미지에 대해 다양한 선 검출 알고리즘을 테스트하고 결과를 저장합니다.

실행 방법 (프로젝트 루트 'samsung2024'에서 실행):
python YOLO_AirLine/test_clone.py ^
  --data_dir "C:\\...\\realDataset_1" ^
  --out_dir  "C:\\...\\line_exp"
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
from collections import defaultdict
from pathlib import Path

# [추가] 메모리 진단을 위한 라이브러리
import gc
import psutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import CRG311 as crg
from deximodel import DexiNed
from abs_6_dof import *
from run_inference import CoordNet, run_inference, run_evaluation
from run_metric import run_formula_based_inference
from pendant_inference import calculate_final_information

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

# ───────── 로컬 모듈 임포트 (중요) ──────────
# 이 스크립트는 상위 디렉토리에서 실행되므로, 
# CRG311.pyd, deximodel.py 등이 있는 경로를 sys.path에 추가해줘야 합니다.
sys.path.append(os.path.join(os.path.dirname(__file__))) # YOLO_AirLine 폴더
sys.path.append(r"C:\Users\user\Desktop\study\task\weld2025\AirLine\build")

# ────────────────── 1. 전역 변수 및 모델 초기화 ─────────────────────
# YOLO 모델
YOLO_MODEL = YOLO(r".\YOLO_AirLine\best_250722.pt")

# AirLine 모델 (Coordinate Detector.py와 완전히 동일한 방식)
THETA_RES, K_SIZE = 6, 9

def build_oridet():
    """OrientationDetector 빌드 함수"""
    thetaN = nn.Conv2d(1, THETA_RES, K_SIZE, 1, K_SIZE // 2, bias=False).cuda()
    for i in range(THETA_RES):
        kernel = np.zeros((K_SIZE, K_SIZE))
        angle = i * 180 / THETA_RES
        x = (np.cos(angle / 180 * np.pi) * (K_SIZE // 2)).astype(np.int32)
        y = (np.sin(angle / 180 * np.pi) * (K_SIZE // 2)).astype(np.int32)
        cv2.line(kernel, (K_SIZE // 2 - x, K_SIZE // 2 - y),
                 (K_SIZE // 2 + x, K_SIZE // 2 + y), 1, 1)
        thetaN.weight.data[i] = torch.tensor(kernel, dtype=torch.float32)
    return thetaN

ORI_DET = build_oridet()
EDGE_DET = DexiNed().cuda()

# [중요] 원본과 동일하게 상위 폴더에서 실행하는 것을 가정하고 모델 경로 설정
# 또한 .eval()을 호출하지 않음으로써 "블랙 매직"을 유지합니다.
edge_state_dict = torch.load(r".\YOLO_AirLine\dexi.pth", map_location='cuda:0')
EDGE_DET.load_state_dict(edge_state_dict)

AL_CFG = dict(edgeThresh=0, simThresh=0.9, pixelNumThresh=10)

# AirLine용 스크래치 버퍼
TMP1 = np.zeros((50000, 2), np.int32)
TMP2 = np.zeros((2, 300000, 2), np.int32)
TMP3 = np.zeros((3000, 2, 2), np.float32)

# 좌표변환 config
STANDALONE_L_FILLET = np.array([909, 1589])
STANDALONE_R_FILLET = np.array([2316, 1573])
STANDALONE_IS_DTS = False

# 예외 처리 파라미터
BLUR_THRESHOLD_1ST = 15.0  # 1차 블러 필터링 임계값
BLUR_THRESHOLD_2ND = 300.0 # 2차 블러 (마커 영역) 필터링 임계값
ROTATION_THRESHOLD_DEG = 50.0 # 허용 총 회전 각도 (pitch+yaw+roll의 절대값 합)

# 6. MLP 모델 경로
MLP_MODEL_PATH = r"C:\Users\user\Desktop\study\task\weld2025\weld2025_samsung_git_temp\testing\samsung2024\welding_project\welding_project\model_A.pth"

# ─────────────────────── 2. 보조 및 처리 함수 ────────────────────────
EXTS = {".png", ".jpg", ".jpeg"}

def laplacian_variance(gray_image: np.ndarray):
    """그레이스케일 이미지의 라플라시안 분산을 계산하여 블러 정도를 측정합니다."""
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def run_pre_flight_checks(bgr_img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, img_name: str) -> bool:
    """
    이미지 처리 전, 블러와 카메라 각도를 검사하여 처리 가능 여부를 반환합니다.
    """
    print(f"    [PRE-CHECK] Pre-flight checks for {img_name}...")
    
    # 1. 1차 전체 이미지 블러 검사
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    var = laplacian_variance(gray_img)
    if var < BLUR_THRESHOLD_1ST:
        print(f"    [FAIL] Image is too blurry (Variance: {var:.2f} < {BLUR_THRESHOLD_1ST}). Skipping.")
        return False
    
    # 2. ArUco 마커 검출
    # [수정] estimate_aruco_pose가 이제 실패 시 None들을 반환하므로, ids만 체크하면 됨.
    rvecs, tvecs, ids, corners = estimate_aruco_pose(bgr_img, camera_matrix, dist_coeffs)
    if ids is None:
        print(f"    [FAIL] ArUco marker not detected. Skipping.")
        return False
        
    # 3. 마커 영역만 잘라내어 2차 블러 검사
    src_pts = corners.astype(np.float32)
    dst_pts = np.array([[0,0],[199,0],[199,199],[0,199]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    marker_crop = cv2.warpPerspective(bgr_img, M, (200,200))
    marker_gray = cv2.cvtColor(marker_crop, cv2.COLOR_BGR2GRAY)
    marker_var = laplacian_variance(marker_gray)
    
    if marker_var < BLUR_THRESHOLD_2ND:
        print(f"    [FAIL] Marker region is too blurry (Variance: {marker_var:.2f} < {BLUR_THRESHOLD_2ND}). Skipping.")
        return False
        
    # 4. 카메라 각도 검사
    R_mat, _ = cv2.Rodrigues(rvecs[0])
    p_y_r = rotation_matrix_to_euler_angles(R_mat)
    p_y_r = np.degrees(p_y_r)
    _, relative_rvecs = convert_marker_relative_6dof_to_camera_relative_6dof(tvecs[0][0], p_y_r) # tvecs가 이중 리스트일 수 있으므로 [0][0] 접근
    
    total_rotation = np.sum(np.abs(relative_rvecs))
    if total_rotation > ROTATION_THRESHOLD_DEG:
        print(f"    [FAIL] Camera angle is too high (Total Rotation: {total_rotation:.2f}° > {ROTATION_THRESHOLD_DEG}°). Skipping.")
        return False
        
    print(f"    [PASS] Pre-flight checks passed.")
    return True

def sharp_S(gray_blur):
    mu = gray_blur.mean()
    varL = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
    return varL / (mu**2 + 1e-6) * 255 / (mu + 12.8)

def enhance(gray_blur, S, t1=2e-3, t2=6e-3, clip_hi=8.0):
    if S < t1:
        clip = clip_hi
    elif S < t2:
        clip = 1 + (clip_hi - 1) * (t2 - S) / (t2 - t1)
    else:
        return cv2.GaussianBlur(gray_blur, (3, 3), 0.8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray_blur)

def enhance_color(bgr_image):
    """
    L*a*b* 색상 공간을 사용하여, 컬러 정보는 유지한 채 밝기 채널만 동적으로 보정합니다.
    """
    # 1. BGR -> L*a*b* 변환 및 채널 분리
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 2. L-Channel(밝기)에 대해 동적 전처리 수행
    l_blur = cv2.GaussianBlur(l_channel, (3, 3), 0.8)
    S = sharp_S(l_blur)
    
    # enhance 함수는 그레이스케일 이미지를 받아 처리하므로 L-channel을 그대로 사용
    enhanced_l = enhance(l_blur, S)

    # 3. 보정된 L 채널과 원본 a, b 채널 재결합
    updated_lab_image = cv2.merge((enhanced_l, a_channel, b_channel))

    # 4. BGR로 다시 변환하여 반환
    return cv2.cvtColor(updated_lab_image, cv2.COLOR_LAB2BGR)

def get_line_pixels(x1, y1, x2, y2):
    """Bresenham's line algorithm과 유사하게 선분을 구성하는 모든 픽셀 좌표 반환"""
    points = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def find_best_fit_line_ransac(lines_by_algo, roi_w, roi_h, consensus_weight=5):
    """
    [단순화] 주어진 선 후보들을 바탕으로, AirLine 중심의 가중 RANSAC을 수행하여 최적의 선 모델을 찾습니다.
    """
    # 1. 모든 후보 선을 하나의 리스트로 통합
    all_candidate_lines = []
    for lines in lines_by_algo.values():
        if lines.size > 0:
            all_candidate_lines.extend(lines)

    # 2. 픽셀 풀 생성 및 가중치 부여
    pixels_by_algo = defaultdict(set)
    for algo, lines in lines_by_algo.items():
        if lines.size == 0: continue
        for x1, y1, x2, y2 in lines:
            pixels_by_algo[algo].update(get_line_pixels(int(x1), int(y1), int(x2), int(y2)))

    # 3. AirLine과 다른 알고리즘들 간의 교차 픽셀(Consensus) 찾기
    airline_pixels = pixels_by_algo.get("AirLine", set())
    if not airline_pixels: # AirLine 결과가 없으면 일반 RANSAC 수행
        all_pixels = set().union(*pixels_by_algo.values())
        if len(all_pixels) < 2: return None
        return run_ransac_on_points(np.array(list(all_pixels)), roi_w, roi_h)

    other_pixels = set()
    for algo, pixels in pixels_by_algo.items():
        if algo != "AirLine": other_pixels.update(pixels)
    
    consensus_pixels = airline_pixels.intersection(other_pixels)
    
    # 4. 가중치 적용을 위한 픽셀 풀 생성
    pixel_pool = []
    pixel_pool.extend(list(airline_pixels - consensus_pixels))
    pixel_pool.extend(list(other_pixels - consensus_pixels))
    pixel_pool.extend(list(consensus_pixels) * consensus_weight)
    
    if len(pixel_pool) < 2: return None
        
    points = np.array(pixel_pool)
    
    # 5. 품질 보증 RANSAC 실행
    ransac = RANSACRegressor(max_trials=10000)
    try:
        x_range = np.ptp(points[:, 0])
        y_range = np.ptp(points[:, 1])
        is_vertical = x_range < y_range
        
        X = points[:, 1].reshape(-1, 1) if is_vertical else points[:, 0].reshape(-1, 1)
        y = points[:, 0] if is_vertical else points[:, 1]
        
        ransac.fit(X, y)
        
        # [수정] 인라이어 비율을 계산하고 출력만 함 (결과 폐기 로직 없음)
        inlier_ratio = np.mean(ransac.inlier_mask_)
        print(f"    [DEBUG]   - RANSAC Inlier Ratio: {inlier_ratio:.2f}")

        # 선 연장 로직
        m, c = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        if is_vertical:
            x1, x2 = m * 0 + c, m * roi_h + c
            y1, y2 = 0, roi_h
        else:
            y1, y2 = m * 0 + c, m * roi_w + c
            x1, x2 = 0, roi_w
        
        extended_line = extend_line((x1, y1), (x2, y2), roi_w, roi_h)
        return extended_line

    except ValueError:
        return None

def find_model_for_warped_lines(lines_by_algo, consensus_weight=5):
    """
    [수정완료] 보정된 공간의 선들을 위한 RANSAC. 
    수직선 (x = c)과 일반선 (y = mx + c)을 정확히 구분하여 반환합니다.
    """
    pixel_pool = []
    pixels_by_algo = defaultdict(set)
    
    for algo, lines in lines_by_algo.items():
        if lines.size > 0:
            for x1, y1, x2, y2 in lines:
                pixels_by_algo[algo].update(get_line_pixels(int(x1), int(y1), int(x2), int(y2)))

    airline_pixels = pixels_by_algo.get("AirLine", set())
    other_pixels = {p for algo, pixels in pixels_by_algo.items() if algo != "AirLine" for p in pixels}
    consensus_pixels = airline_pixels.intersection(other_pixels)

    pixel_pool.extend(list(airline_pixels - consensus_pixels))
    pixel_pool.extend(list(other_pixels - consensus_pixels))
    pixel_pool.extend(list(consensus_pixels) * consensus_weight)

    if len(pixel_pool) < 2:
        return None
    points = np.array(pixel_pool)

    ransac = RANSACRegressor(max_trials=10000)

    try:
        y_range = np.ptp(points[:, 1])
        x_range = np.ptp(points[:, 0])
        is_vertical = y_range > x_range

        if is_vertical:
            x_mean = np.mean(points[:, 0])
            return 0, x_mean, True  # 수직선은 m=0으로 지정
        else:
            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]
            ransac.fit(X, y)
            m = ransac.estimator_.coef_[0]
            c = ransac.estimator_.intercept_
            return m, c, False  # 일반선 모델 (y=mx+c)

    except ValueError:
        return None


def filter_lines_by_diagonal(lines: np.ndarray, roi_w: int, roi_h: int):
    """
    ROI의 대각선 각도 '콘(cone)'을 기준으로, 선들을 수평/수직 지향으로 필터링합니다.
    """
    if lines.size == 0:
        return {'horizontal': np.empty((0, 4)), 'vertical': np.empty((0, 4))}

    # 1. 대각선 기준 각도 계산 (0-90도)
    diag_angle_deg = np.rad2deg(np.arctan2(roi_h, roi_w))

    # 2. 각 선분의 각도 계산 (0-180도)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    # arctan2는 -180~180 범위를 반환하므로, 0~180 범위로 변환
    line_angles_deg = np.rad2deg(np.mod(np.arctan2(dy, dx), np.pi))

    # 3. '나비넥타이' 또는 '모래시계' 모양의 필터 마스크 생성
    # 수평 콘 필터: (0 ~ diag_angle) U (180-diag_angle ~ 180)
    horizontal_mask = (line_angles_deg <= diag_angle_deg) | \
                      (line_angles_deg >= (180 - diag_angle_deg))
    
    # 수직 콘 필터: (90-diag_angle) ~ (90+diag_angle)
    vertical_mask = (line_angles_deg >= (90 - diag_angle_deg)) & \
                    (line_angles_deg <= (90 + diag_angle_deg))

    return {
        'horizontal': lines[horizontal_mask],
        'vertical': lines[vertical_mask]
    }

def run_ransac_on_points(points: np.ndarray, roi_w: int, roi_h: int):
    """주어진 점들에 대해 RANSAC을 실행하고 연장된 선분 좌표를 반환합니다."""
    if len(points) < 2:
        return None

    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    try:
        x_range = np.ptp(X)
        y_range = np.ptp(y)

        if x_range < y_range:
            ransac = RANSACRegressor().fit(y.reshape(-1, 1), X.ravel())
            is_vertical = True
        else:
            ransac = RANSACRegressor().fit(X, y)
            is_vertical = False
    except ValueError:
        return None

    if is_vertical:
        m, c = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        x1, x2 = m * 0 + c, m * roi_h + c
        y1, y2 = 0, roi_h
    else:
        m, c = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
        y1, y2 = m * 0 + c, m * roi_w + c
        x1, x2 = 0, roi_w

    points_to_clip = []
    # top/bottom
    for t_y in [0, roi_h]:
        if y1 != y2:
            x = x1 + (x2 - x1) * (t_y - y1) / (y2 - y1)
            if 0 <= x <= roi_w: points_to_clip.append((int(x), t_y))
    # left/right
    for t_x in [0, roi_w]:
        if x1 != x2:
            y = y1 + (y2 - y1) * (t_x - x1) / (x2 - x1)
            if 0 <= y <= roi_h: points_to_clip.append((t_x, int(y)))
    
    if len(points_to_clip) < 2: return None

    # 중복점 제거 (코너에서 만날 경우)
    unique_points = sorted(list(set(points_to_clip)))
    if len(unique_points) < 2: return None
    
    return unique_points[0], unique_points[-1]

def extend_line(p1, p2, roi_w, roi_h):
    """
    [신설] 두 점으로 정의된 선분을 ROI 경계까지 연장합니다.
    RANSAC의 무작위성 없이, 결정론적으로 동작합니다.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # 수직선, 수평선 등 엣지 케이스 처리
    if x1 == x2: # 수직선
        return (x1, 0), (x1, roi_h)
    if y1 == y2: # 수평선
        return (0, y1), (roi_w, y1)

    # 직선의 방정식 y = mx + c 계산
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    # 경계와 만나는 교점 계산
    points = []
    # y = 0 (상단)
    x = -c / m
    if 0 <= x <= roi_w: points.append((x, 0))
    # y = roi_h (하단)
    x = (roi_h - c) / m
    if 0 <= x <= roi_w: points.append((x, roi_h))
    # x = 0 (좌측)
    y = c
    if 0 <= y <= roi_h: points.append((0, y))
    # x = roi_w (우측)
    y = m * roi_w + c
    if 0 <= y <= roi_h: points.append((roi_w, y))
    
    # 유효한 교점이 2개 이상일 때만 반환
    if len(points) >= 2:
        # 중복 제거 및 정렬하여 양 끝점을 반환
        unique_points = sorted(list(set(points)))
        p_start = (int(round(unique_points[0][0])), int(round(unique_points[0][1])))
        p_end = (int(round(unique_points[-1][0])), int(round(unique_points[-1][1])))
        return p_start, p_end
    
    return None

def filter_line_by_centrality(lines, roi_w, roi_h, tolerance_ratio=0.10):
    """
    [개선] 주어진 선분들 중, 무한 연장선이 ROI 중심 영역을 통과하는 것들만 필터링합니다.
    """
    if lines.size == 0:
        return np.empty((0, 4))

    center_x, center_y = roi_w / 2, roi_h / 2
    radius = min(roi_w, roi_h) * tolerance_ratio
    
    passed_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        
        # 선과 점 사이의 최단 거리 계산 (직선 기준)
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([center_x, center_y])
        
        distance = np.abs(np.cross(p2 - p1, p1 - p3)) / (np.linalg.norm(p2 - p1) + 1e-6)
        
        if distance <= radius:
            passed_lines.append(line)
            
    return np.array(passed_lines) if passed_lines else np.empty((0, 4))

# ────────────────── 3. 선 검출 알고리즘 래퍼 ──────────────────
def run_lsd(img):
    det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = det.detect(img)[0]
    if lines is None:
        return np.empty((0, 4))
    
    # [추가] LSD 결과에 대한 길이 필터 후처리
    min_len_sq = 40**2
    long_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x2 - x1)**2 + (y2 - y1)**2 > min_len_sq:
            long_lines.append([x1, y1, x2, y2])
            
    return np.array(long_lines) if long_lines else np.empty((0, 4))

def run_fld(img):
    # [수정] 기준 재조정
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=25)
    l = fld.detect(img)
    return np.empty((0, 4)) if l is None else l.reshape(-1, 4)

def run_hough(img):
    e = cv2.Canny(img, 40, 120)
    # [수정] 최종 파라미터 조정
    l = cv2.HoughLinesP(e, 1, np.pi / 180, 30, 40, 20)
    return np.empty((0, 4)) if l is None else l.reshape(-1, 4)

def run_airline(roi_bgr: np.ndarray, airline_config: dict):
    """성공이 확인된 AirLine 로직을 함수로 래핑"""
    H0, W0 = roi_bgr.shape[:2]
    
    res = 16; dscale = 1
    resized_height = H0 // dscale // res * res
    resized_width = W0 // dscale // res * res
    if resized_height == 0 or resized_width == 0:
        return np.empty((0, 4), float)
        
    rx1_resized = cv2.resize(roi_bgr, (resized_width, resized_height))
    scale_x = W0 / resized_width
    scale_y = H0 / resized_height

    rx1_resized = np.ascontiguousarray(rx1_resized)
    x1 = torch.tensor(rx1_resized, dtype=torch.float32).cuda() / 255.0
    x1 = x1.permute(2, 0, 1)

    with torch.no_grad():
        edgeDetection = EDGE_DET(x1.unsqueeze(0))

    edgeNp = edgeDetection.detach().cpu().numpy()[0, 0]
    
    edgeNp_binary = (edgeNp > airline_config["edgeThresh"]).astype(np.uint8) * 255
    if np.sum(edgeNp_binary) == 0:
        return np.empty((0, 4), float)

    ODes = ORI_DET(edgeDetection)
    ODes = torch.nn.functional.normalize(ODes - ODes.mean(1, keepdim=True), p=2.0, dim=1)

    outMap = np.zeros_like(edgeNp, dtype=np.uint8)
    outMap = np.expand_dims(outMap, 2).repeat(3, axis=2)
    out = np.zeros((3000, 2, 3), dtype=np.float32)

    rawLineNum = crg.desGrow(
        outMap, edgeNp_binary, ODes[0].detach().cpu().numpy(), out,
        airline_config["simThresh"], airline_config["pixelNumThresh"],
        TMP1, TMP2, TMP3, THETA_RES
    )
    rawLineNum = min(rawLineNum, 3000)

    if rawLineNum == 0: return np.empty((0, 4), float)

    lines = []
    for i in range(rawLineNum):
        pt1 = (out[i, 0, 1] * scale_x, out[i, 0, 0] * scale_y)
        pt2 = (out[i, 1, 1] * scale_x, out[i, 1, 0] * scale_y)
        lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    
    return np.asarray(lines, float)

ALGOS = {
    "LSD": (run_lsd, (0, 255, 0)),
    "FLD": (run_fld, (0, 0, 255)),
    "Hough": (run_hough, (255, 0, 0)),
    "AirLine": (run_airline, (0, 255, 255)),
}

# ─────────────────────── 4. 메인 루프 ──────────────────────────────
def get_intersection(line1, line2):
    """ 두 선분의 교점을 계산합니다. """
    if line1 is None or line2 is None:
        return None
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 선이 평행하거나 동일함

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # 교점이 선분 내에 있을 때만 유효함 (지금은 무한 직선의 교점을 찾음)
    # if not (0 <= t <= 1 and 0 <= u <= 1):
    #     return None

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return int(ix), int(iy)

def process(data_dir: Path, out_root: Path, airline_config: dict):
    """
    주어진 데이터셋과 AirLine 설정을 사용하여 선 검출을 수행하고 결과를 저장합니다.
    """
    # [추가] 카메라 파라미터 로드
    try:
        camera_matrix = np.load('./YOLO_AirLine/pose_estimation_code_and_camera_matrix/camera_parameters/camera_matrix_filtered.npy')
        dist_coeffs = np.load('./YOLO_AirLine/pose_estimation_code_and_camera_matrix/camera_parameters/dist_coeffs_filtered.npy')
        print("[INFO] Real camera parameters loaded.")
    except FileNotFoundError:
        print("[ERROR] Camera parameter files not found! Pose estimation will be skipped.")
        camera_matrix, dist_coeffs = None, None

    # [수정] 새로운 데이터셋 구조에 맞게 경로 처리 로직 단순화
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    if not image_dir.exists():
        print(f"[오류] 이미지 폴더를 찾을 수 없습니다: {image_dir.resolve()}")
        return
        
    imgs = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in EXTS])
    
    print(f"\n[INFO] 현재 설정: {airline_config}")
    print(f"총 {len(imgs)} 장의 이미지를 처리합니다.")

    # [수정] YOLO 결과 및 JSON, 호모그래피, 포즈추정 시각화 결과 저장 폴더 추가
    for n in list(ALGOS.keys()) + ["RANSAC_combined", "enhanced_full_image", "Collar_Filter_Debug", "Welding_Filter_Debug", "YOLO_only_results", "final_json_results", "Homography_Results", "Pose_Estimation_Vis"]:
        (out_root / n).mkdir(parents=True, exist_ok=True)
    
    for i, p in enumerate(tqdm(imgs, desc="Total Progress")):
        print(f"\n[INFO] >> 이미지 처리 시작 ({i+1}/{len(imgs)}): {p.name}")
        
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  [WARN] 이미지를 읽을 수 없어 건너뜁니다.")
            continue

        # [추가] 예외 처리 검사 수행
        if camera_matrix is not None and dist_coeffs is not None:
            # 새로 추가된 함수들을 사용하기 위해 bgr 이미지를 전달합니다.
            is_ok = run_pre_flight_checks(bgr, camera_matrix, dist_coeffs, p.name)
            if not is_ok:
                # 검사에 실패하면 해당 이미지는 건너뜀
                # 실패 이유는 함수 내부에서 출력됨
                continue

        # [추가] 최종 JSON 출력을 위한 데이터 구조 초기화
        output_data = {
            "is_collar": 0, "is_collar_hole": 0, "collar_type": 0, "is_hole_left": 0, "is_hole_right": 0,
            "hole_left_bounding_width": 0, "hole_right_bounding_width": 0,
            "coordinates": {"longi_left_upper_x": 0, "longi_left_upper_y": 0, "longi_left_lower_x": 0, "longi_left_lower_y": 0,
                            "longi_right_upper_x": 0, "longi_right_upper_y": 0, "longi_right_lower_x": 0, "longi_right_lower_y": 0,
                            "collar_left_lower_x": 0, "collar_left_lower_y": 0, "collar_left_upper_x": 0, "collar_left_upper_y": 0},
            "pixel_scalar": {"fillet": 0, "longi": 0, "collar_vertical": 0, "collar_horizontal": 0},
            "camera_pose": {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "roll": 0}
        }
        
        acc = {n: [] for n in ALGOS}
        ransac_vis = bgr.copy()
        enhanced_image_canvas = bgr.copy()
        collar_debug_vis = bgr.copy()
        welding_debug_vis = bgr.copy()

        h, w = bgr.shape[:2]
        
        # [수정] 1. YOLO 결과 분석 및 JSON 특징 추출 (1단계) ---
        yolo_results = YOLO_MODEL(bgr, verbose=False, conf=0.6)
        
        # [수정] 순수 YOLO 결과 시각화 및 저장 (향상된 가시성)
        yolo_only_vis = bgr.copy()
        if yolo_results and yolo_results[0].boxes:
            # 폰트, 굵기, 크기 설정
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                
                # 클래스별 색상 정의 (BGR 순서)
                if cls == 0:
                    color = (203, 192, 255)  # 핑크색
                elif cls == 1:
                    color = (255, 0, 0)      # 파란색
                elif cls == 2:
                    color = (0, 0, 255)      # 빨간색
                elif cls in [3, 4, 5, 6]:
                    color = (0, 255, 0)      # 녹색
                else:
                    color = (255, 255, 255)  # 기본 흰색
                    
                # Bbox 그리기
                cv2.rectangle(yolo_only_vis, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨 텍스트 생성 및 그리기
                label = f'{YOLO_MODEL.names[cls]} {conf:.2f}'
                (label_width, label_height), baseline = cv2.getTextSize(label, font_face, font_scale, thickness)
                
                # 텍스트 배경 사각형 그리기
                cv2.rectangle(yolo_only_vis, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
                # 텍스트 쓰기 (흰색)
                cv2.putText(yolo_only_vis, label, (x1, y1 - baseline), font_face, font_scale, (255, 255, 255), 2)
                
        cv2.imwrite(str(out_root / "YOLO_only_results" / f"{p.stem}_yolo.png"), yolo_only_vis)

        rois = []
        boxes_by_class = defaultdict(list)
        if yolo_results and yolo_results[0].boxes:
            for box in yolo_results[0].boxes:
                class_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rois.append((class_id, x1, y1, x2, y2))
                boxes_by_class[class_id].append([x1, y1, x2, y2])
        
        # [핵심 수정] ROI 리스트를 클래스 ID 기준으로 정렬
        rois.sort(key=lambda x: x[0])

        # [추가] YOLO 기반 특징 계산
        collar_classes = {3, 4, 5, 6}
        detected_collar_classes = collar_classes.intersection(boxes_by_class.keys())
        if detected_collar_classes:
            output_data["is_collar"] = 1
            # 여러 칼라 클래스가 검출될 경우 첫 번째 것을 기준으로 collar_type 설정
            first_collar_cls = sorted(list(detected_collar_classes))[0]
            if first_collar_cls == 3: output_data["collar_type"] = 3
            elif first_collar_cls == 4: output_data["collar_type"] = 0
            elif first_collar_cls == 5: output_data["collar_type"] = 1
            elif first_collar_cls == 6: output_data["collar_type"] = 2

        hole_boxes = boxes_by_class.get(0, [])
        collar_boxes = [box for cls in detected_collar_classes for box in boxes_by_class[cls]]

        if hole_boxes and collar_boxes:
            for h_box in hole_boxes:
                for c_box in collar_boxes:
                    # 두 박스의 IoU(Intersection over Union)를 계산하여 겹치는지 확인
                    hx1, hy1, hx2, hy2 = h_box
                    cx1, cy1, cx2, cy2 = c_box
                    inter_x1 = max(hx1, cx1)
                    inter_y1 = max(hy1, cy1)
                    inter_x2 = min(hx2, cx2)
                    inter_y2 = min(hy2, cy2)
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        output_data["is_collar_hole"] = 1
                        break
                if output_data["is_collar_hole"] == 1:
                    break
        
        for h_box in hole_boxes:
            h_center_x = (h_box[0] + h_box[2]) / 2
            h_width = h_box[2] - h_box[0]
            if h_center_x < w / 2:
                output_data["is_hole_left"] = 1
                output_data["hole_left_bounding_width"] = h_width
            else:
                output_data["is_hole_right"] = 1
                output_data["hole_right_bounding_width"] = h_width

        # [추가] ROI 처리 결과를 담을 리스트
        processed_results = []

        # ROI가 없으면 전체 이미지를 대상으로 (기존 로직 유지)
        if not rois:
            rois = [(0, 0, 0, w, h)]

        # [신규] 1. 루프 시작 전, 선행 처리로 H 행렬 계산
        H = None
        guideline_rois = [r for r in rois if r[0] in [1, 2]]
        temp_guideline_results = {}
        for cls_h, x1_h, y1_h, x2_h, y2_h in guideline_rois:
            roi_bgr_h = bgr[y1_h:y2_h, x1_h:x2_h]
            if roi_bgr_h.size == 0: continue
            enhanced_h = enhance_color(roi_bgr_h)
            lines_by_algo_h = {n: fn(cv2.cvtColor(enhanced_h, cv2.COLOR_BGR2GRAY)) if "air" not in n.lower() else fn(enhanced_h, airline_config) for n, (fn, _) in ALGOS.items()}
            ransac_line_h = find_best_fit_line_ransac(lines_by_algo_h, roi_bgr_h.shape[1], roi_bgr_h.shape[0])
            if ransac_line_h:
                p1_g = (int(ransac_line_h[0][0] + x1_h), int(ransac_line_h[0][1] + y1_h))
                p2_g = (int(ransac_line_h[1][0] + x1_h), int(ransac_line_h[1][1] + y1_h))
                line_info_h = {"line": [p1_g[0], p1_g[1], p2_g[0], p2_g[1]]}
                if cls_h == 1: temp_guideline_results['fillet'] = line_info_h
                else:
                    if (x1_h + x2_h) / 2 < w / 2: temp_guideline_results['longi_left'] = line_info_h
                    else: temp_guideline_results['longi_right'] = line_info_h
        
        if 'fillet' in temp_guideline_results and 'longi_left' in temp_guideline_results and 'longi_right' in temp_guideline_results:
            l_pt = get_intersection(temp_guideline_results['fillet']['line'], temp_guideline_results['longi_left']['line'])
            r_pt = get_intersection(temp_guideline_results['fillet']['line'], temp_guideline_results['longi_right']['line'])
            if l_pt and r_pt:
                try:
                    _, _, H = get_absolute_pose(bgr, camera_matrix, dist_coeffs, np.array(l_pt), np.array(r_pt), is_dts=False)
                except Exception: H = None # 실패 시 H는 None
        if H is not None: print("  [INFO] Master Homography Matrix H calculated successfully.")

        processed_results = []
        left_longi_res = None
        right_longi_res = None
        fillet_res = None # fillet_res도 함께 초기화
        
        # [신규] 2. 메인 루프 실행
        for rid, (cls, x1_roi, y1_roi, x2_roi, y2_roi) in enumerate(rois):
            # [추가] 클래스 0 처리 로직
            if cls == 0:
                print(f"    [INFO] Class 0 ROI 발견, 바운딩 박스만 표시합니다.")
                cv2.rectangle(ransac_vis, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 255, 255), 1)
                continue

            roi_bgr = bgr[y1_roi:y2_roi, x1_roi:x2_roi]
            if roi_bgr.size == 0: continue
            
            # --- 1. 전처리 ---
            roi_bgr_enhanced = enhance_color(roi_bgr)
            roi_gray_enhanced = cv2.cvtColor(roi_bgr_enhanced, cv2.COLOR_BGR2GRAY)

            # --- 2. 모든 알고리즘으로 선 검출 ---
            lines_by_algo = {}
            for n, (fn, _) in ALGOS.items():
                if n == "AirLine":
                    lines_by_algo[n] = run_airline(roi_bgr_enhanced, airline_config)
                else:
                    lines_by_algo[n] = fn(roi_gray_enhanced)

            # --- 3. [중요] 필터링 전의 원본 결과를 먼저 저장용으로 누적 ---
            for n, lines in lines_by_algo.items():
                lines_global = lines.copy()
                if lines_global.size:
                    lines_global[:, [0, 2]] += x1_roi
                    lines_global[:, [1, 3]] += y1_roi
                acc[n].append(lines_global)

            # --- 4. RANSAC 및 필터링 ---
            # 클래스 1, 2 먼저 처리
            if cls in [1, 2]:
                # 1. 방향성 필터
                filtered_after_diag = {}
                for algo, lines in lines_by_algo.items():
                    result_dict = filter_lines_by_diagonal(lines, roi_bgr.shape[1], roi_bgr.shape[0])
                    key = 'horizontal' if roi_bgr.shape[1] > roi_bgr.shape[0] else 'vertical'
                    filtered_after_diag[algo] = result_dict[key]

                # 2. "지능형 재시도" 중심 영역 필터
                all_passed_diag_list = [lines for lines in filtered_after_diag.values() if lines.size > 0]
                passed_after_center = np.empty((0,4), dtype=np.float32)

                if all_passed_diag_list:
                    all_passed_diag = np.vstack(all_passed_diag_list)
                    
                    for tolerance in [0.10, 0.20, 0.30, 0.40, 0.50]:
                        passed_after_center = filter_line_by_centrality(all_passed_diag, roi_bgr.shape[1], roi_bgr.shape[0], tolerance_ratio=tolerance)
                        if passed_after_center.size > 0:
                            break
                
                # 3. 가중 RANSAC을 위한 최종 후보 재구성
                final_candidates = defaultdict(list)
                if passed_after_center.size > 0:
                    passed_center_set = {tuple(map(int, row)) for row in passed_after_center}
                    for algo, lines in filtered_after_diag.items():
                        if lines.size > 0:
                            for line in lines:
                                if tuple(map(int, line)) in passed_center_set:
                                    final_candidates[algo].append(line)
                
                for algo, lines in final_candidates.items():
                    final_candidates[algo] = np.array(lines)
                
                print(f"    [DEBUG]   - RANSAC 후보 (가중치 적용 대상):")
                for algo, lines in final_candidates.items():
                    print(f"      - {algo}: {len(lines)}개 선")

                ransac_line = find_best_fit_line_ransac(final_candidates, roi_bgr.shape[1], roi_bgr.shape[0])
                
                if ransac_line:
                    pt1, pt2 = ransac_line
                    pt1_global = (int(pt1[0] + x1_roi), int(pt1[1] + y1_roi))
                    pt2_global = (int(pt2[0] + x1_roi), int(pt2[1] + y1_roi))
                    
                    current_res = {
                        "class_id": cls,
                        "type": "ransac_line",
                        "line": [pt1_global[0], pt1_global[1], pt2_global[0], pt2_global[1]],
                        "roi": [x1_roi, y1_roi, x2_roi, y2_roi],
                        "airline_lines_in_roi": lines_by_algo.get("AirLine", np.empty((0,4)))
                    }
                    
                    processed_results.append(current_res)

                    if cls == 2:
                        if (x1_roi + x2_roi)/2 < w/2:
                            left_longi_res = current_res
                        else:
                            right_longi_res = current_res

                    col = {1: (255, 0, 0), 2: (0, 0, 255)}.get(cls, (255,0,255))
                    cv2.line(ransac_vis, pt1_global, pt2_global, col, 3)

            # 이후 클래스 3, 4, 5, 6 처리
            elif cls in [3, 4, 5, 6]:
                is_left = (x1_roi + x2_roi) / 2 < w / 2
                three_sides_roi = find_3_sides_of_collar(lines_by_algo, roi_bgr.shape[1], roi_bgr.shape[0], is_left)

                # 여기서 전역 좌표로 변환 (ROI 좌상단을 더해줌)
                three_sides = {}
                for key, line in three_sides_roi.items():
                    if line is not None:
                        line_global = [line[0] + x1_roi, line[1] + y1_roi, line[2] + x1_roi, line[3] + y1_roi]
                        three_sides[key] = line_global

                longi_to_use = left_longi_res if is_left else right_longi_res

                print(f"    [DEBUG] Checking conditions: len(three_sides) = {len(three_sides)}, longi_to_use is not None: {longi_to_use is not None}")

                if len(three_sides) == 3 and longi_to_use:
                    p1 = get_intersection(three_sides['top'], three_sides['outer_vertical'])
                    p2 = get_intersection(three_sides['bottom'], three_sides['outer_vertical'])
                    p3 = get_intersection(three_sides['top'], longi_to_use['line'])
                    p4 = get_intersection(three_sides['bottom'], longi_to_use['line'])

                    final_points = [p for p in [p1, p2, p3, p4] if p is not None]

                    if len(final_points) == 4:
                        final_points = sort_rectangle_points_clockwise(final_points)

                    if len(final_points) == 4:
                        processed_results.append({
                            "class_id": cls,
                            "type": "collar_points",
                            "points": final_points
                        })
                        cv2.polylines(ransac_vis, [np.array(final_points, dtype=np.int32)], True, (0,255,0), 3)        
        # --- 2. 좌표 계산 및 JSON 특징 추출 (2단계) ---
        fillet_line = None
        longi_lines = {"left": [], "right": []}
        collar_item = None
        
        # 1. 처리된 결과 분류
        for res in processed_results:
            if res["class_id"] == 1 and res["type"] == "ransac_line":
                fillet_line = res["line"]
            elif res["class_id"] == 2 and res["type"] == "ransac_line":
                # ROI의 중심 x좌표를 기준으로 좌/우 구분
                roi_center_x = (res["roi"][0] + res["roi"][2]) / 2
                if roi_center_x < w / 2:
                    longi_lines["left"].append(res)
                else:
                    longi_lines["right"].append(res)
            elif res["type"] == "collar_points":
                collar_item = res

        # 2. 좌표 계산
        coords = output_data["coordinates"]
        
        # 클래스 2가 좌/우 하나씩 있다고 가정하고 첫번째 것만 사용
        left_longi_res = longi_lines["left"][0] if longi_lines["left"] else None
        right_longi_res = longi_lines["right"][0] if longi_lines["right"] else None

        if fillet_line and left_longi_res:
            pt = get_intersection(fillet_line, left_longi_res["line"])
            if pt: coords["longi_left_lower_x"], coords["longi_left_lower_y"] = pt
        
        if fillet_line and right_longi_res:
            pt = get_intersection(fillet_line, right_longi_res["line"])
            if pt: coords["longi_right_lower_x"], coords["longi_right_lower_y"] = pt

        # [최종 수정] Longi가 없을 경우, Hole ROI와 '이차 곡선 감속'으로 좌표 추정
        if fillet_line:
            fillet_dx = fillet_line[2] - fillet_line[0]
            fillet_dy = fillet_line[3] - fillet_line[1]
            # 각도를 0(수평)~1(수직) 범위로 정규화
            normalized_angle = np.arctan2(np.abs(fillet_dy), np.abs(fillet_dx)) / (np.pi / 2)
            
            # [수정] "이차 이징(Quadratic Easing)"을 사용하여 각도에 따른 오프셋 감소를 가속화
            power = 0.5
            interpolation_factor = (1 - normalized_angle) ** power

            # 왼쪽 Longi 없고 왼쪽 Hole 있을 때
            if coords["longi_left_lower_x"] == 0 and output_data["is_hole_left"] == 1:
                left_hole_boxes = [b for b in boxes_by_class.get(0, []) if (b[0] + b[2]) / 2 < w / 2]
                if left_hole_boxes:
                    hole_box = min(left_hole_boxes, key=lambda b: b[0])
                    
                    min_offset = (hole_box[2] - hole_box[0]) * 0.3
                    max_offset = (hole_box[2] - hole_box[0]) * 0.9
                    dynamic_offset = min_offset + (max_offset - min_offset) * interpolation_factor
                    
                    # [핵심 수정] 홀의 '오른쪽(안쪽)' 경계에서 안쪽으로 오프셋을 뺌
                    virtual_x = int(hole_box[2] - dynamic_offset)
                    virtual_line = [virtual_x, 0, virtual_x, h]
                    
                    pt = get_intersection(fillet_line, virtual_line)
                    if pt:
                        coords["longi_left_lower_x"], coords["longi_left_lower_y"] = pt
                        print(f"  [INFO] Left lower point ESTIMATED INWARDLY at x={virtual_x}")

            # 오른쪽 Longi 없고 오른쪽 Hole 있을 때
            if coords["longi_right_lower_x"] == 0 and output_data["is_hole_right"] == 1:
                right_hole_boxes = [b for b in boxes_by_class.get(0, []) if (b[0] + b[2]) / 2 >= w / 2]
                if right_hole_boxes:
                    hole_box = max(right_hole_boxes, key=lambda b: b[2])

                    min_offset = (hole_box[2] - hole_box[0]) * 0.3
                    max_offset = (hole_box[2] - hole_box[0]) * 0.9
                    dynamic_offset = min_offset + (max_offset - min_offset) * interpolation_factor
                    
                    # [핵심 수정] 홀의 '왼쪽(안쪽)' 경계에서 안쪽으로 오프셋을 더함
                    virtual_x = int(hole_box[0] + dynamic_offset)
                    virtual_line = [virtual_x, 0, virtual_x, h]
                    
                    pt = get_intersection(fillet_line, virtual_line)
                    if pt:
                        coords["longi_right_lower_x"], coords["longi_right_lower_y"] = pt
                        print(f"  [INFO] Right lower point ESTIMATED INWARDLY at x={virtual_x}")

        # [수정] 계층적 Upper Point 탐색 로직 -> 수선의 발 전략 폐기하고 단순화
        upper_pt_left = find_upper_point_by_intersection(left_longi_res)
        if not upper_pt_left:
            upper_pt_left = find_upper_point_by_roi_fallback(left_longi_res)

        upper_pt_right = find_upper_point_by_intersection(right_longi_res)
        if not upper_pt_right:
            upper_pt_right = find_upper_point_by_roi_fallback(right_longi_res)

        if upper_pt_left: coords["longi_left_upper_x"], coords["longi_left_upper_y"] = upper_pt_left
        if upper_pt_right: coords["longi_right_upper_x"], coords["longi_right_upper_y"] = upper_pt_right
        
        # [수정] Collar Point 탐색 로직 -> 위치 기반으로 단순화
        if collar_item:
            points = np.array(collar_item["points"])
            
            if len(points) == 4:
                # 4개 교점의 x좌표 중심 계산
                collar_center_x = np.mean(points[:, 0])
                
                # x좌표 기준으로 점들을 정렬
                sorted_by_x = sorted(points, key=lambda p: p[0])

                target_points = []
                # 칼라가 이미지 왼쪽에 있으면, x값이 큰 두 점 (오른쪽 변)을 선택
                if collar_center_x < w / 2:
                    target_points = sorted_by_x[2:]
                # 칼라가 이미지 오른쪽에 있으면, x값이 작은 두 점 (왼쪽 변)을 선택
                else:
                    target_points = sorted_by_x[:2]

                if len(target_points) == 2:
                    p_a, p_b = target_points[0], target_points[1]
    
                    # y값에 따라 upper, lower 결정
                    if p_a[1] < p_b[1]:
                        coords["collar_left_upper_x"], coords["collar_left_upper_y"] = int(p_a[0]), int(p_a[1])
                        coords["collar_left_lower_x"], coords["collar_left_lower_y"] = int(p_b[0]), int(p_b[1])
                    else:
                        coords["collar_left_upper_x"], coords["collar_left_upper_y"] = int(p_b[0]), int(p_b[1])
                        coords["collar_left_lower_x"], coords["collar_left_lower_y"] = int(p_a[0]), int(p_a[1])
        
        # [추가] 호모그래피 적용 전의 원본 좌표를 시각화용으로 백업
        points_to_draw_orig = {
            "L_LOWER": (coords["longi_left_lower_x"], coords["longi_left_lower_y"]),
            "R_LOWER": (coords["longi_right_lower_x"], coords["longi_right_lower_y"]),
            "L_UPPER": (coords["longi_left_upper_x"], coords["longi_left_upper_y"]),
            "R_UPPER": (coords["longi_right_upper_x"], coords["longi_right_upper_y"]),
            "C_LOWER": (coords["collar_left_lower_x"], coords["collar_left_lower_y"]),
            "C_UPPER": (coords["collar_left_upper_x"], coords["collar_left_upper_y"]),
        }

        # --- 3. 6-DoF 자세 추정 ---
        if camera_matrix is not None and dist_coeffs is not None:
            l_fillet_pt = np.array([coords["longi_left_lower_x"], coords["longi_left_lower_y"]], dtype='int32')
            r_fillet_pt = np.array([coords["longi_right_lower_x"], coords["longi_right_lower_y"]], dtype='int32')

            # 필렛 포인트가 유효할 때만 자세 추정 시도
            if np.all(l_fillet_pt != 0) and np.all(r_fillet_pt != 0):
                print("  [INFO] Attempting to estimate camera pose...")
                try:
                    # [수정] Pose Estimation 시각화 저장을 위해 save_dir, img_path 인자 전달
                    pose_vis_save_dir = out_root / "Pose_Estimation_Vis"
                    abs_tvecs, abs_rvecs, H = get_absolute_pose(bgr, camera_matrix, dist_coeffs, l_fillet_pt, r_fillet_pt, is_dts=False, save_dir=pose_vis_save_dir, img_path=p)
                    
                    if abs_tvecs is None: # 마커 검출 실패 등의 이유로 None이 반환된 경우
                        print("  [WARN] get_absolute_pose returned None. Skipping further pose processing.")
                        continue

                    output_data["camera_pose"]["x"] = float(abs_tvecs[0])
                    output_data["camera_pose"]["y"] = float(abs_tvecs[1])
                    output_data["camera_pose"]["z"] = float(abs_tvecs[2])
                    output_data["camera_pose"]["pitch"] = float(abs_rvecs[0])
                    output_data["camera_pose"]["yaw"] = float(abs_rvecs[1])
                    output_data["camera_pose"]["roll"] = float(abs_rvecs[2])
                    print("  [INFO] Pose estimation successful.")

                    # 호모그래피 적용
                    if H is not None:
                        # 1. 변환할 모든 좌표를 (N, 1, 2) 형태의 numpy 배열로 만듦
                        original_points = []
                        keys = []
                        for key, value in coords.items():
                            if key.endswith('_x'):
                                y_key = key.replace('_x', '_y')
                                if coords[y_key] != 0:
                                    original_points.append([[coords[key], coords[y_key]]])
                                    keys.append(key.replace('_x', ''))

                        if original_points:
                            original_points_np = np.array(original_points, dtype=np.float32)
                            transformed_points_np = cv2.perspectiveTransform(original_points_np, H)
                            
                            for i, base_key in enumerate(keys):
                                trans_pt = transformed_points_np[i][0]
                                coords[base_key + '_x'] = int(trans_pt[0])
                                coords[base_key + '_y'] = int(trans_pt[1])
                            print("  [INFO] Homography applied to all coordinates.")
                        
                        # [추가] 호모그래피 적용된 이미지 및 좌표 시각화
                        warped_image = cv2.warpPerspective(bgr, H, (w, h))
                        transformed_points_to_draw = {
                            "L_LOWER": (coords["longi_left_lower_x"], coords["longi_left_lower_y"]),
                            "R_LOWER": (coords["longi_right_lower_x"], coords["longi_right_lower_y"]),
                            "L_UPPER": (coords["longi_left_upper_x"], coords["longi_left_upper_y"]),
                            "R_UPPER": (coords["longi_right_upper_x"], coords["longi_right_upper_y"]),
                            "C_LOWER": (coords["collar_left_lower_x"], coords["collar_left_lower_y"]),
                            "C_UPPER": (coords["collar_left_upper_x"], coords["collar_left_upper_y"]),
                        }
                        for label, (px, py) in transformed_points_to_draw.items():
                             if px != 0 and py != 0:
                                color = (0, 0, 255)
                                if "UPPER" in label: color = (255, 100, 0)
                                if "C_" in label: color = (0, 255, 0)
                                cv2.circle(warped_image, (px, py), radius=10, color=color, thickness=-1)
                                cv2.putText(warped_image, label, (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        cv2.imwrite(str(out_root / "Homography_Results" / f"{p.stem}_homography.png"), warped_image)
                        print("  [INFO] Saved homography-transformed image with points.")

                        # --- 5. Pixel Scalar 계산 ---
                        print("  [INFO] Calculating pixel scalars from transformed coordinates...")
                        
                        # Fillet 길이
                        if coords["longi_right_lower_x"] != 0 and coords["longi_left_lower_x"] != 0:
                            output_data["pixel_scalar"]["fillet"] = float(abs(coords["longi_right_lower_x"] - coords["longi_left_lower_x"]))

                        # Longi 길이
                        longi_lengths = []
                        if coords["longi_left_upper_y"] != 0 and coords["longi_left_lower_y"] != 0:
                            longi_lengths.append(abs(coords["longi_left_upper_y"] - coords["longi_left_lower_y"]))
                        if coords["longi_right_upper_y"] != 0 and coords["longi_right_lower_y"] != 0:
                            longi_lengths.append(abs(coords["longi_right_upper_y"] - coords["longi_right_lower_y"]))
                        
                        if longi_lengths:
                            output_data["pixel_scalar"]["longi"] = float(np.mean(longi_lengths))

                        # Collar 길이
                        if coords["collar_left_upper_x"] != 0:
                            if collar_item and 'points' in collar_item:
                                collar_pts_orig = np.array(collar_item['points'], dtype=np.float32).reshape(-1, 1, 2)
                                collar_pts_trans = cv2.perspectiveTransform(collar_pts_orig, H)
                                
                                x_coords = collar_pts_trans[:, 0, 0]
                                y_coords = collar_pts_trans[:, 0, 1]
                                
                                width = np.max(x_coords) - np.min(x_coords)
                                height = np.max(y_coords) - np.min(y_coords)

                                output_data["pixel_scalar"]["collar_horizontal"] = float(width)
                                output_data["pixel_scalar"]["collar_vertical"] = float(height)

                except Exception as e:
                    print(f"  [WARN] Pose estimation failed: {e}")
            else:
                print("  [INFO] Fillet points not found, skipping pose estimation.")

        # [삭제] 최종 교점 시각화 로직 (위로 이동했음)
        
        # --- 이미지 루프 끝난 후 파일 저장 ---
        #cv2.imwrite(str(out_root / "Collar_Filter_Debug" / f"{p.stem}_collar_debug.png"), collar_debug_vis)
        cv2.imwrite(str(out_root / "Welding_Filter_Debug" / f"{p.stem}_welding_debug.png"), welding_debug_vis)

        # [추가] RANSAC_combined 저장 전에 백업해둔 원본 좌표로 점 그리기
        for label, (px, py) in points_to_draw_orig.items():
            if px != 0 and py != 0:
                color = (0, 0, 255)  # Default Red (LOWER)
                if "UPPER" in label: color = (255, 100, 0)  # Blue
                if "C_" in label: color = (0, 255, 0)      # Green (Collar)
                cv2.circle(ransac_vis, (px, py), radius=10, color=color, thickness=-1)
                cv2.putText(ransac_vis, label, (px + 15, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imwrite(str(out_root / "RANSAC_combined" / f"{p.stem}_ransac.png"), ransac_vis)
        cv2.imwrite(str(out_root / "enhanced_full_image" / f"{p.stem}_enhanced.jpg"), enhanced_image_canvas)
        
        # [추가] JSON 파일 저장
        json_path = out_root / "final_json_results" / f"{p.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        # [수정] AirLine 결과 저장 시 클래스별 색상 적용
        for n, ( _, base_col) in ALGOS.items():
             vis = bgr.copy()
             all_lines_for_algo = [ln for ln in acc[n] if ln.size > 0]
             if not all_lines_for_algo:
                 # 선이 없어도 빈 이미지는 저장 (일관성을 위해)
                 cv2.imwrite(str(out_root / n / f"{p.stem}_{n}.png"), vis)
                 continue
             
             for rid_save, lines_save in enumerate(acc[n]):
                 if lines_save.size == 0: continue
                 
                 # ROI 리스트에 클래스가 없는 경우를 대비한 방어 코드0
                 if rid_save < len(rois):
                     cls_save = rois[rid_save][0]
                 else:
                     cls_save = -1 # 기본값 또는 에러 처리

                 # AirLine 폴더에 저장할 때만 클래스별 색상 적용, 나머지는 기본색
                 if n == "AirLine":
                     col = {1: (255, 0, 0), 2: (0, 0, 255)}.get(cls_save, (0, 255, 0) if cls_save in [3,4,5,6] else base_col)
                 else:
                     col = base_col
                 for lx1, ly1, lx2, ly2 in lines_save.astype(int):
                     cv2.line(vis, (lx1, ly1), (lx2, ly2), col, 3)
             
             cv2.imwrite(str(out_root / n / f"{p.stem}_{n}.png"), vis)
             
             final_lines_json = np.vstack(all_lines_for_algo)
             with open(out_root/ n /f"{p.stem}_{n}.json", "w") as f:
                 json.dump(final_lines_json.tolist(), f)

        # [추가] 메모리 정리 및 상태 출력
        process_mem = psutil.Process(os.getpid())
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [MEMORY] - CPU: {process_mem.memory_info().rss / 1024 ** 2:.1f} MB | GPU: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB (Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB)")


def find_upper_point_by_intersection(longi_res):
    """AirLine 수직 교차선을 이용해 Upper Point를 찾습니다."""
    if not longi_res: return None
            
    longi_line = np.array(longi_res["line"])
    roi = longi_res["roi"]
    x1_r, y1_r, x2_r, y2_r = roi
    airline_lines_roi = longi_res["airline_lines_in_roi"]
    
    if airline_lines_roi.size == 0: return None

    longi_dx = longi_line[2] - longi_line[0]
    longi_dy = longi_line[3] - longi_line[1]
    longi_angle = np.rad2deg(np.mod(np.arctan2(longi_dy, longi_dx), np.pi))
    
    candidate_intersections = []
    # [추가] ROI 높이 및 상위 영역 기준 설정
    roi_height = y2_r - y1_r
    upper_region_y_threshold = roi_height * 0.15 # 상위 15%

    for al_line in airline_lines_roi:
        al_dx = al_line[2] - al_line[0]
        al_dy = al_line[3] - al_line[1]
        al_angle = np.rad2deg(np.mod(np.arctan2(al_dy, al_dx), np.pi))
        
        angle_diff = abs(longi_angle - al_angle)
        if abs(angle_diff - 90) < 15:
            midpoint_y = (al_line[1] + al_line[3]) / 2
            if midpoint_y < upper_region_y_threshold:
                al_line_global = al_line + [x1_r, y1_r, x1_r, y1_r]
                pt = get_intersection(longi_line, al_line_global)
                if pt:
                    candidate_intersections.append(pt)
    
    if candidate_intersections:
        return min(candidate_intersections, key=lambda p: p[1])
    return None

def find_upper_point_by_roi_fallback(longi_res):
    """ROI 경계를 기준으로 Upper Point 폴백 로직을 수행합니다."""
    if not longi_res: return None
    longi_line = longi_res["line"]
    x1_r, y1_r, x2_r, y2_r = longi_res["roi"]
    lx1, ly1, lx2, ly2 = longi_line
    
    # User's Tweak: y_intersect + height * 0.035
    offset = (y2_r - y1_r) * 0.035

    if (lx2 - lx1) == 0:
        return (int(lx1), int(y1_r + offset))
    else:
        m = (ly2 - ly1) / (lx2 - lx1)
        c = ly1 - m * lx1
        y_intersect = y1_r
        x_intersect = (y_intersect - c) / m
        return (int(x_intersect), int(y_intersect + offset))

def find_3_sides_of_collar(lines_by_algo, roi_w, roi_h, is_left_side):
    airline_lines = lines_by_algo.get("AirLine")
    if airline_lines is None or airline_lines.size == 0:
        return {}

    airline_lines = np.clip(airline_lines, [0, 0, 0, 0], [roi_w, roi_h, roi_w, roi_h])

    directional_groups = filter_lines_by_diagonal(airline_lines, roi_w, roi_h)
    h_lines = directional_groups['horizontal']
    v_lines = directional_groups['vertical']

    # Top과 Bottom 후보 선정
    top_lines = [l for l in h_lines if max(l[1], l[3]) < roi_h * 0.2]
    bottom_lines = [l for l in h_lines if min(l[1], l[3]) > roi_h * 0.8]

    print(f"[DEBUG] Top 후보 개수: {len(top_lines)}개")
    for i, l in enumerate(top_lines):
        length = np.hypot(l[2]-l[0], l[3]-l[1])
        print(f"  [Top 후보 {i}] 좌표: {l}, 길이: {length:.2f}")

    print(f"[DEBUG] Bottom 후보 개수: {len(bottom_lines)}개")
    for i, l in enumerate(bottom_lines):
        length = np.hypot(l[2]-l[0], l[3]-l[1])
        print(f"  [Bottom 후보 {i}] 좌표: {l}, 길이: {length:.2f}")

    # 길이와 연결성을 고려한 최종 선정
    top_line = max(top_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), -max(l[1], l[3])), default=None)
    bottom_line = max(bottom_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), min(l[1], l[3])), default=None)

    if top_line is not None:
        print(f"[DEBUG] 최종 선택된 Top: {top_line}, 길이: {np.hypot(top_line[2]-top_line[0], top_line[3]-top_line[1]):.2f}")
    if bottom_line is not None:
        print(f"[DEBUG] 최종 선택된 Bottom: {bottom_line}, 길이: {np.hypot(bottom_line[2]-bottom_line[0], bottom_line[3]-bottom_line[1]):.2f}")

    outer_v_line = None
    if len(v_lines) > 0:
        if is_left_side:
            right_lines = [l for l in v_lines if min(l[0], l[2]) > roi_w * 0.7]
            outer_v_line = max(right_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), (l[0]+l[2])/2), default=None)
            print(f"[DEBUG] Right(Outer Vertical) 후보 개수: {len(right_lines)}개")
            for i, l in enumerate(right_lines):
                mid_x = (l[0]+l[2])/2
                length = np.hypot(l[2]-l[0], l[3]-l[1])
                print(f"  [Right 후보 {i}] 좌표: {l}, 중점x: {mid_x:.2f}, 길이: {length:.2f}")
        else:
            left_lines = [l for l in v_lines if max(l[0], l[2]) < roi_w * 0.3]
            outer_v_line = max(left_lines, key=lambda l: (np.hypot(l[2]-l[0], l[3]-l[1]), -(l[0]+l[2])/2), default=None)
            print(f"[DEBUG] Left(Outer Vertical) 후보 개수: {len(left_lines)}개")
            for i, l in enumerate(left_lines):
                mid_x = (l[0]+l[2])/2
                length = np.hypot(l[2]-l[0], l[3]-l[1])
                print(f"  [Left 후보 {i}] 좌표: {l}, 중점x: {mid_x:.2f}, 길이: {length:.2f}")

    if outer_v_line is not None:
        print(f"[DEBUG] 최종 선택된 Outer Vertical: {outer_v_line}")

    found_edges = {}
    if top_line is not None: found_edges['top'] = top_line
    if bottom_line is not None: found_edges['bottom'] = bottom_line
    if outer_v_line is not None: found_edges['outer_vertical'] = outer_v_line
        
    return found_edges

def sort_rectangle_points_clockwise(pts):
    """
    네 점을 시계방향으로 정렬합니다.
    좌상단 점에서 시작하여 시계방향으로 순서를 만듭니다.
    """
    pts = np.array(pts)
    # 중심점 계산
    center = np.mean(pts, axis=0)

    # 각 점의 중심점 대비 각도 계산
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])

    # 각도가 작은 순서대로 정렬 (반시계방향)
    sort_idx = np.argsort(angles)

    # 시계방향으로 바꾸려면 순서를 뒤집음
    sorted_pts = pts[sort_idx[::-1]]

    return sorted_pts.astype(int).tolist()
# ─────────────────────── 5. CLI 및 실험 실행기 ────────────────────────
if __name__ == "__main__":
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    if not os.path.exists('YOLO_AirLine'):
        print("오류: 이 스크립트는 프로젝트 루트(samsung2024)에서 실행해야 합니다.")
        sys.exit(1)

    ap = argparse.ArgumentParser(description="선 검출 알고리즘 파라미터 조합 테스트")
    ap.add_argument("--data_dir", required=True, help="데이터셋 루트 (내부에 image, labels 폴더 존재)")
    ap.add_argument("--out_dir", default="./exp_lines", help="결과를 저장할 루트 폴더")
    args = ap.parse_args()

    # --- 실험 파라미터 셋 정의 ---
    param_sets = []
    px_thresholds = [300] # 300의 성능이 가장 좋음
    # (edgeThresh, simThresh) 조합
    # [수정] 테스트할 전략만 남기고 주석 처리
    strategies = {
        # "Strict":                   (1.0, 0.98),
        # "Generous":                 (-1.0, 0.85),
        "QuantityFocused":          (-1.0, 0.98),
        #"QualityFocused":           (1.0, 0.85),
        "QualityFocused_GenerousSim": (1.0, 0.75), 
    }

    for name, (edge, sim) in strategies.items():
        for px in px_thresholds:
            param_sets.append({
                "name": f"strategy={name}_px={px}",
                "config": {"edgeThresh": edge, "simThresh": sim, "pixelNumThresh": px}
            })

    # --- 실험 실행 ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = Path(f"{args.out_dir}")
    print(f"총 {len(param_sets)}개의 파라미터 셋으로 실험을 시작합니다.")
    print(f"결과는 '{base_out_dir}' 폴더에 저장됩니다.")

    for param in tqdm(param_sets, desc="Total Progress"):
        # 각 파라미터 셋을 위한 별도의 출력 폴더 생성
        out_path_for_set = base_out_dir / param["name"]
        process(Path(args.data_dir), out_path_for_set, param["config"])

    print("모든 실험이 완료되었습니다.")

    # --- [추가] 후처리: 추론 및 평가 실행 ---
    print("\n--- 모든 실험 완료. 추론 및 평가를 시작합니다. ---")
    
    # # MLP 모델 로드
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = CoordNet().to(device)
    # try:
    #     state = torch.load(MLP_MODEL_PATH, map_location=device)
    #     model.load_state_dict(state)
    #     model.eval()
    #     print(f"추론 모델 로드 성공: {MLP_MODEL_PATH}")
    # except FileNotFoundError:
    #     print(f"[오류] 추론 모델 파일을 찾을 수 없습니다: {MLP_MODEL_PATH}. 후처리를 건너뜁니다.")
    #     sys.exit(1)
    
    # 각 실험 결과에 대해 추론/평가 실행
    for param in param_sets:
        out_path_for_set = base_out_dir / param["name"]
        print(f"\n>>> '{param['name']}' 결과에 대한 후처리 시작...")
        
        inference_input_dir = out_path_for_set / "final_json_results"
        inference_result_dir = out_path_for_set / "inference_results"
        
        if not inference_input_dir.is_dir() or not any(inference_input_dir.iterdir()):
            print(f"  [경고] 추론할 JSON 파일이 없거나 폴더를 찾을 수 없습니다. 건너뜁니다: {inference_input_dir}")
            continue
            
        # 추론 실행
        run_formula_based_inference(str(inference_input_dir), str(inference_result_dir))
        calculate_final_information(str(inference_input_dir), str(inference_result_dir))
        # 평가 실행 (라벨 폴더가 있는 경우)
        labels_dir = Path(args.data_dir) / "labels"
        if labels_dir.is_dir():
            run_evaluation(str(inference_result_dir), str(labels_dir))
        else:
            print(f"  [정보] 라벨 폴더를 찾을 수 없어 평가를 건너뜁니다: {labels_dir}")
            
    print("\n--- 모든 후처리 작업이 완료되었습니다. ---")