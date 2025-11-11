import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


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

    # 2) DetectorParameters
    try:
        aruco_params = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        # create() 메서드가 없으면 기본 생성자로 대체
        aruco_params = cv2.aruco.DetectorParameters()

    # 2. 마커 검출
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    corners[0][0] = reorder_corners_by_position(corners[0][0])

    if ids is None:
        print("No markers detected")
        return image, [], [], []

    # 3. 마커 pose 추정
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs
    )

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

def get_absolute_pose(img, camera_matrix, dist_coeffs, l_fillet_point, r_fillet_point, is_dts=False):
    image = img.copy()

    rvecs, tvecs, _, corners = estimate_aruco_pose(
        image,
        camera_matrix,
        dist_coeffs,
        marker_length=0.1  # 마커 한 변의 길이 (미터 단위)
    )

    # Rotation Vector to Euler Angles
    R, _ = cv2.Rodrigues(rvecs[0])
    p_y_r = rotation_matrix_to_euler_angles(R)
    p_y_r = np.degrees(p_y_r)

    # 카메라 기준 상대 좌표계로 변환
    relative_tvecs, relative_rvecs = convert_marker_relative_6dof_to_camera_relative_6dof(tvecs, p_y_r)
    relative_tvecs = relative_tvecs[0][0] 

    marker_center = np.mean(corners, axis=0)
    height, width = image.shape[:2]

    marker_pixels = 200
    if is_dts:
        marker_pixels = 50

    angles = np.arctan2(corners[:, 1] - marker_center[1],
                    corners[:, 0] - marker_center[0])

    # 각도를 기준으로 코너 정렬 (가장 작은 각도가 우상단)
    sorted_indices = np.argsort(angles)
    marker_corners = corners[sorted_indices]

    # 우상단 코너를 기준으로 시계방향으로 회전하여 정렬
    rotation_offset = (4 - np.where(angles == np.min(angles))[0][0]) % 4
    marker_corners = np.roll(marker_corners, rotation_offset, axis=0)

    # === 기준 사각형: 마커 기준 정렬된 평면 좌표
    # 실제 단위 필요 없음, 비율만 맞으면 OK
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

    img = warped_full.copy()

    l_fillet_warped_point = cv2.perspectiveTransform(np.array([[l_fillet_point]], dtype='float32'), H)[0][0].astype(int)
    r_fillet_warped_point = cv2.perspectiveTransform(np.array([[r_fillet_point]], dtype='float32'), H)[0][0].astype(int)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H10)
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict)
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
    abs_marker_center *= 10.0 / marker_pixels  # cm 단위로 변환

    abs_marker_vector = y_flip_matrix @ abs_rotation_matrix @ marker_vector

    val = np.dot(marker_vector, fillet_vector) / (np.linalg.norm(marker_vector) * np.linalg.norm(fillet_vector))
    theta = np.arccos(val)
    if abs_marker_vector[1] < 0:
        theta_deg = -np.degrees(theta)
    else:
        theta_deg = np.degrees(theta)


    absolute_rvecs = np.array([relative_rvecs[0], relative_rvecs[1], relative_rvecs[2] + theta_deg])
    absolute_tvecs = np.array([relative_tvecs[0] - 0.01 * abs_marker_center[0], relative_tvecs[1] + 0.01 * abs_marker_center[1], relative_tvecs[2]])
    print(f'absolute_tvecs(m): {absolute_tvecs}')
    print(f'absolute_rvecs(deg): {absolute_rvecs}')

    return absolute_tvecs, absolute_rvecs


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
    raw_img = cv2.imread(img_path)


    # --------INPUT-------                                               -------OUTPUT(Camera 6DoF)-------
    #    raw_img:       전처리되지 않은 현장 촬영 raw 이미지                  absolute_tvecs: (3,) 크기의 numpy array. 필렛 중앙으로부터 카메라의 x, y, z 좌표(m단위)
    #    camera_matrix: 카메라 내부 파라미터                                absoulte_rvecs: (3,) 크기의 numpy array. 필렛 중앙으로부터 카메라의 pitch, yaw, roll(호도법 단위)
    #    dist_coeffs:   렌즈 왜곡 계수
    #    is_dts:        DTS 이미지 여부
    abs_translation, abs_rotation = get_absolute_pose(raw_img, camera_matrix, dist_coeffs, l_fillet_point, r_fillet_point, is_dts)


if __name__=='__main__':
    main()