import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import os

def draw_axis_and_print(img, K, dist, rvec, tvec, length=0.1):
    """
    이미지에 좌표축(X, Y, Z)과 각 축 끝점, 마커 중심의 좌표(cam frame 기준)를 시각화하고 출력함

    Parameters:
        img (ndarray): 입력 이미지
        K (ndarray): 카메라 내재 행렬
        
          (ndarray): 왜곡 계수
        rvec (ndarray): 회전 벡터
        tvec (ndarray): 이동 벡터
        length (float): 각 축 길이 (미터 단위)

    Returns:
        cam_coords (ndarray): 카메라 좌표계 기준의 X, Y, Z 축 끝점 (3x3)
    """
    # 1) 축 끝점 3D (마커 기준)
    axis_3d = np.float32([
        [length, 0, 0],   # X
        [0, length, 0],   # Y
        [0, 0, length]    # Z
    ]).reshape(-1, 3)
    labels = ['X', 'Y', 'Z']
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR

    # 2) 3D→2D 투영 (픽셀 좌표)
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    # 3) 회전행렬 변환 → 카메라 좌표계상의 위치 계산
    R, _ = cv2.Rodrigues(rvec)
    cam_coords = (R @ axis_3d.T + tvec.reshape(3, 1)).T  # shape: (3, 3)

    # 4) 마커 중심(원점) 픽셀 및 3D 좌표
    origin_px, _ = cv2.projectPoints(
        np.zeros((1, 3), np.float32), rvec, tvec, K, dist
    )
    ox, oy = origin_px.ravel().astype(int)
    origin_coord = tvec.ravel()

    cv2.circle(img, (ox, oy), 5, (255, 255, 255), -1)
    cv2.putText(img, f"Origin_cam:({origin_coord[0]:.3f},{origin_coord[1]:.3f},{origin_coord[2]:.3f})m",
                (ox + 5, oy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 각 축 그리기 + 좌표 표시
    for pt, lbl, col, cam_c in zip(imgpts, labels, colors, cam_coords):
        pt = tuple(pt)
        cv2.line(img, (ox, oy), pt, col, 2)

        # 픽셀 좌표 텍스트
        cv2.putText(img, f"{lbl}_px:({pt[0]},{pt[1]})",
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # 3D 좌표 텍스트 (카메라 좌표계)
        cv2.putText(img, f"{lbl}_cam:({cam_c[0]:.3f},{cam_c[1]:.3f},{cam_c[2]:.3f})m",
                    (pt[0] + 5, pt[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # 콘솔 출력
        # print(f"{lbl}-axis endpoint in camera frame [m]: {cam_c}")
        # print(f"거리: {np.linalg.norm(cam_c):.3f}m")

    cv2.imwrite("pose_estimation_annotated_image.png", img)  # 결과 이미지 저장
    return cam_coords[0], cam_coords[1], cam_coords[2]  # X, Y, Z 끝점

def reorder_corners_by_position(pts):
    """
    pts: (4,2) numpy array
    return: reordered (4,2) points in order:
        top-left, top-right, bottom-right, bottom-left
    """
    pts = np.array(pts, dtype=np.float32)

    # 중심점 기준
    center = np.mean(pts, axis=0)

    # 사분면 각도로 정렬
    def angle_from_center(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])

    sorted_pts = sorted(pts, key=angle_from_center)

    # 가장 위쪽(최소 y), 왼쪽(최소 x) 조합을 기준점으로 회전
    top_left_idx = np.argmin([p[0] + p[1] for p in sorted_pts])
    sorted_pts = sorted_pts[top_left_idx:] + sorted_pts[:top_left_idx]

    return np.array(sorted_pts, dtype=np.float32)


def estimate_aruco_pose(image, camera_matrix, dist_coeffs, marker_length=0.1):
    """
    Aruco 마커의 거리 및 회전 정보를 추정하는 함수

    Parameters:
    image (ndarray): 입력 이미지 
    camera_matrix (ndarray): 카메라 내부 파라미터
    dist_coeffs (ndarray): 왜곡 계수
    marker_length (float): 마커 한 변의 길이 (미터 단위)

    Returns:
    annotated_image (ndarray): 마커와 좌표축이 그려진 이미지
    rvecs (list of ndarray): 마커의 회전 벡터 목록
    tvecs (list of ndarray): 마커의 거리(이동) 벡터 목록
    ids (list of int): 감지된 마커 ID 목록
    """

    # 1. ArUco dict 및 parameters 설정
    try:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H10)
    except AttributeError:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H10)

    try:
        aruco_params = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters()

    # 2. 마커 검출
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    
    # [수정] 마커를 찾지 못했으면 IndexError 방지를 위해 즉시 None 반환
    if ids is None or len(corners) == 0:
        return None, None, None, None

    corners[0][0] = reorder_corners_by_position(corners[0][0])

    # 3. 마커 pose 추정
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs
    )

    # [수정] 반환값 형식 일치
    return rvecs, tvecs, ids, corners[0][0]


def convert_marker_relative_6dof_to_camera_relative_6dof(tvecs, p_y_r):
    # tvecs 자체를 마커 기준 카메라의 상대좌표로 바로 활용 가능
    # +x: 오른쪽, +y: 위쪽, +z: 앞쪽
    relative_tvecs = tvecs

    # pitch yaw roll을 마커 기준 카메라의 회전으로
    relative_rvecs = p_y_r
    if p_y_r[0] > 0:
        relative_rvecs[0] = p_y_r[0] - 180
    else:
        relative_rvecs[0] = 180 + p_y_r[0]

    relative_rvecs[1] = -p_y_r[1]

    return relative_tvecs, relative_rvecs


def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # \phi = \arctan2(R_{3,2}, R_{3,3})
        y = np.arcsin(-R[2, 0])           # \theta = \arcsin(-R_{3,1})
        z = np.arctan2(R[1, 0], R[0, 0])  # \psi = \arctan2(R_{2,1}, R_{1,1})
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arcsin(-R[2, 0])
        z = 0
    return np.array([x, y, z])

def get_absolute_pose(img, camera_matrix, dist_coeffs, l_fillet_point, r_fillet_point, is_dts=False, save_dir=None, img_path=None):
    image = img.copy()
    cpy_image = img.copy()

    # [수정] estimate_aruco_pose에서 마커 미검출 시 ValueError가 발생하므로 try-except로 감쌈
    try:
        rvecs, tvecs, _, corners = estimate_aruco_pose(
            image,
            camera_matrix,
            dist_coeffs,
            marker_length=0.1
        )
    except ValueError:
        # 마커 검출 실패 시 None을 반환하여 상위 호출자가 처리하도록 함
        return None, None, None
    
    # Rotation Vector to Euler Angles
    R, _ = cv2.Rodrigues(rvecs[0])
    p_y_r = rotation_matrix_to_euler_angles(R)
    p_y_r = np.degrees(p_y_r)

    tmp_tvecs = tvecs[0][0]
    cam_in_marker = - R.T @ tmp_tvecs

    relative_tvecs, relative_rvecs = convert_marker_relative_6dof_to_camera_relative_6dof(cam_in_marker, p_y_r)

    marker_center = np.mean(corners, axis=0)
    height, width = image.shape[:2]
    marker_pixels = 200
    if is_dts:
        marker_pixels = 50

    angles = np.arctan2(corners[:, 1] - marker_center[1], corners[:, 0] - marker_center[0])
    sorted_indices = np.argsort(angles)
    marker_corners = corners[sorted_indices]
    rotation_offset = (4 - np.where(angles == np.min(angles))[0][0]) % 4
    marker_corners = np.roll(marker_corners, rotation_offset, axis=0)

    dst_pts = np.array([
        [0, 0],
        [marker_pixels, 0],
        [marker_pixels, marker_pixels],
        [0, marker_pixels]
    ], dtype='float32')
    dst_pts += np.array([width // 2 - marker_pixels / 2, height // 2 - marker_pixels / 2], dtype='float32')

    H, _ = cv2.findHomography(marker_corners, dst_pts)
    warped_full = cv2.warpPerspective(image, H, (width, height))
    img = warped_full.copy()

    l_fillet_warped_point = cv2.perspectiveTransform(np.array([[l_fillet_point]], dtype='float32'), H)[0][0].astype(int)
    r_fillet_warped_point = cv2.perspectiveTransform(np.array([[r_fillet_point]], dtype='float32'), H)[0][0].astype(int)

    # ⬇️ pose 시각화 저장
    if img_path is not None and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(save_dir, f"{base}_pose_estimation.png")

        vis = img.copy() # 시각화용 이미지 복사
        try:
            h_rvecs, h_tvecs, h_ids, _ = estimate_aruco_pose(
                img, camera_matrix, dist_coeffs, marker_length=0.1
            )

            # 마커 검출 성공 시에만 축 그리기
            for i in range(len(h_ids)):
                draw_axis_and_print(
                    vis,
                    camera_matrix,
                    dist_coeffs,
                    h_rvecs[i],
                    h_tvecs[i],
                    length=0.1
                )
        except ValueError:
            # 검출 실패 시 경고 출력하고 축 그리기만 건너뜀
            print(f"  [WARNING] (Visualization) ArUco marker not found in warped image for {os.path.basename(img_path)}. Saving without axis.")
        
        cv2.imwrite(out_path, vis) # 마커를 찾았든 못찾았든 시각화 이미지는 저장

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H10)
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
    
    # [수정] 마커 탐지 실패 시 None을 반환하여 프로그램 중단 방지
    if ids is None:
        print(f"  [WARNING] (Calculation) ArUco marker not found in warped image. Cannot calculate absolute pose.")
        return None, None, None

    corners[0][0] = reorder_corners_by_position(corners[0][0])
    corners = corners[0][0]

    right_point = r_fillet_warped_point
    left_point = l_fillet_warped_point
    center_point = (l_fillet_warped_point + r_fillet_warped_point) / 2

    marker_vector = [corners[2][0] - corners[3][0], corners[2][1] - corners[3][1]]
    fillet_vector = [right_point[0] - left_point[0], right_point[1] - left_point[1]]
    theta_fillet = -math.atan2(fillet_vector[1], fillet_vector[0])

    abs_rotation_matrix = np.array([
        [math.cos(theta_fillet), -math.sin(theta_fillet)],
        [math.sin(theta_fillet), math.cos(theta_fillet)]
    ])
    y_flip_matrix = np.array([
        [1, 0],
        [0, -1]
    ])

    marker_center = np.mean(corners, axis=0)
    abs_marker_center = y_flip_matrix @ abs_rotation_matrix @ (marker_center - center_point)
    abs_marker_center *= 10.0 / marker_pixels

    abs_marker_vector = y_flip_matrix @ abs_rotation_matrix @ marker_vector
    val = np.dot(marker_vector, fillet_vector) / (np.linalg.norm(marker_vector) * np.linalg.norm(fillet_vector))
    theta = np.arccos(val)
    theta_deg = -np.degrees(theta) if abs_marker_vector[1] < 0 else np.degrees(theta)

    absolute_rvecs = np.array([relative_rvecs[0], relative_rvecs[1], relative_rvecs[2] + theta_deg])
    absolute_tvecs = np.array([
        relative_tvecs[0] - 0.01 * abs_marker_center[0],
        relative_tvecs[1] + 0.01 * abs_marker_center[1],
        relative_tvecs[2]
    ])

    print(f'absolute_tvecs(m): {absolute_tvecs}')
    print(f'absolute_rvecs(deg): {absolute_rvecs}')
    p_y_r[2] += theta_deg

    image, H = apply_homography_refined(cpy_image, camera_matrix, dist_coeffs)
    # cv2.imwrite(f"./homo_img.jpg", image)

    return absolute_tvecs, absolute_rvecs, H


def main():
    # dts 이미지를 사용하는 경우 True, 실제 현장 이미지를 사용하는 경우 False
    is_dts = False

    if is_dts:
        # dts image 설정
        camera_matrix = np.load('./camera_parameters/unity_intrinsic_matrix.npy')
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 왜곡 계수 초기화

    else:
        # real image 설정
        camera_matrix = np.load('./camera_parameters/camera_matrix_filtered.npy')
        dist_coeffs = np.load('./camera_parameters/dist_coeffs_filtered.npy')


    # 도출된 필렛 좌표 대입
    l_fillet_point = np.array([909, 1589], dtype='int32')
    r_fillet_point = np.array([2316, 1573], dtype='int32')

    img_path = './test_pics_2/5.jpg'
    # 현장 사진 전처리되지 않은 raw 이미지 입력
    raw_img = cv2.imread(r".\YOLO_AirLine\WIN_20250605_10_06_59_Pro.jpg")


    # --------INPUT-------                                               -------OUTPUT(Camera 6DoF)-------
    #    raw_img:       전처리되지 않은 현장 촬영 raw 이미지                  absolute_tvecs: (3,) 크기의 numpy array. 필렛 중앙으로부터 카메라의 x, y, z 좌표(m단위)
    #    camera_matrix: 카메라 내부 파라미터                                absoulte_rvecs: (3,) 크기의 numpy array. 필렛 중앙으로부터 카메라의 pitch, yaw, roll(호도법 단위)
    #    dist_coeffs:   렌즈 왜곡 계수
    #    is_dts:        DTS 이미지 여부
    abs_translation, abs_rotation, _ = get_absolute_pose(raw_img, camera_matrix, dist_coeffs, l_fillet_point, r_fillet_point, is_dts)

def apply_homography(image, camera_matrix, dist_coeffs, marker_pixels=200):
    height, width = image.shape[:2]
    # 렌즈 왜곡 보정
    image = cv2.undistort(image, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # === ArUco 마커 검출 ===
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H10)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is None or len(corners) < 1:
        raise ValueError("No ArUco markers detected!")
    marker_corners = corners[0][0]
    marker_corners = reorder_corners_by_position(marker_corners)
    center = np.mean(marker_corners, axis=0)
    angles = np.arctan2(marker_corners[:, 1] - center[1],
                    marker_corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    marker_corners = marker_corners[sorted_indices]
    rotation_offset = (4 - np.where(angles == np.min(angles))[0][0]) % 4
    marker_corners = np.roll(marker_corners, rotation_offset, axis=0)
    # === 기준 사각형: 마커 기준 정렬된 평면 좌표
    dst_pts = np.array([
        [0, 0],       # TL
        [marker_pixels, 0],     # TR
        [marker_pixels, marker_pixels],   # BR
        [0, marker_pixels]      # BL
    ], dtype='float32')
    dst_pts += np.array([width // 2 - marker_pixels / 2, height // 2 - marker_pixels / 2], dtype='float32')
    # === Homography 계산: 마커 → 기준 사각형
    H, _ = cv2.findHomography(marker_corners, dst_pts)
    # === Homography를 원본 해상도에 적용
    # size=(width, height) 유지
    warped_full = cv2.warpPerspective(image, H, (width, height))
    # save warped_full
    # cv2.imwrite(f"./homo_{pic_num}.jpg", warped_full)
    # === 결과 시각화
    # cv2.imshow("Original", image)
    resized_wraped_img = cv2.resize(warped_full, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow("Homography Applied (Full Image)", resized_wraped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return warped_full, H

def apply_homography_refined(image, camera_matrix, dist_coeffs, marker_pixels=200):
    h, w = image.shape[:2]
    # 1) 왜곡 보정 및 그레이
    image = cv2.undistort(image, camera_matrix, dist_coeffs)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2) 마커 검출
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36H10
    )
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is None or len(corners) == 0:
        raise ValueError("No ArUco markers detected!")
    # ────────────────────────────────────────
    # 3) 코너 서브픽셀 refine (튜플 → 리스트로 수집)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)
    refined = []
    for c in corners:
        # c: (4,1,2) float32
        c_ref = cv2.cornerSubPix(
            gray, c, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
        )
        refined.append(c_ref)
    corners = refined  # 이제 corners 는 list[ndarray]
    # ────────────────────────────────────────
    # 4) 첫 마커 코너 정렬
    marker_corners = reorder_corners_by_position(corners[0].reshape(4, 2))
    center  = marker_corners.mean(axis=0)
    angles  = np.arctan2(marker_corners[:, 1] - center[1],
                         marker_corners[:, 0] - center[0])
    marker_corners = marker_corners[np.argsort(angles)]
    rot = (4 - np.argmin(angles)) % 4
    marker_corners = np.roll(marker_corners, rot, axis=0)
    # 5) 기준 사각형 좌표
    dst_pts = np.array([[0, 0],
                        [marker_pixels, 0],
                        [marker_pixels, marker_pixels],
                        [0, marker_pixels]], dtype=np.float32)
    dst_pts += np.array([w/2 - marker_pixels/2,
                         h/2 - marker_pixels/2], dtype=np.float32)
    # 6) Homography 계산 & 적용
    H, _ = cv2.findHomography(marker_corners, dst_pts)
    warped = cv2.warpPerspective(image, H, (w, h))
    return warped, H

if __name__=='__main__':
    main() 