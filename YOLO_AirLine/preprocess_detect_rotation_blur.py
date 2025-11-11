#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import glob
import os

### CONSTANTS ###
BLUR_THRESHOLD_1ST = 15 # 블러 판단을 위한 임계값
BLUR_THRESHOLD_2ND = 300 # 차후 수정
camera_matrix = np.load('./camera_parameters/camera_matrix_filtered.npy')
dist_coeffs = np.load('./camera_parameters/dist_coeffs_filtered.npy')


# **Blur Detection**

# In[7]:


def laplacian_variance(image_path):
    # 이미지 읽기 (그레이스케일)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 라플라시안 변환 적용
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    
    return laplacian_var


# **Relative 6DoF**

# In[ ]:


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

    if ids is None:
        print("No markers detected")
        return image, [], [], []

    # 2. 마커 검출
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    corners[0][0] = reorder_corners_by_position(corners[0][0])


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


# **Detect Blur and Rotation**

# In[ ]:


def detect_blur_and_rotation(image):
    var = laplacian_variance(image)

    # 1차 필터링
    if var < BLUR_THRESHOLD_1ST:
        Warning(f'Image {os.path.relpath(image)} is blurred, skipping further processing.')
        return False
    else:
        print(f'Image {os.path.relpath(image)} passed the first blur check.')

    rvecs, tvecs, _, corners = estimate_aruco_pose(cv2.imread(image), camera_matrix, dist_coeffs, marker_length=0.1)

    # corners가 없으면 패스
    if corners is None or len(corners) != 4:
        print(f'No marker detected in {os.path.relpath(image)}, skipping.\n')
        return False

    # 마커 부분만 perspective transform으로 crop
    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array([[0,0],[199,0],[199,199],[0,199]], dtype=np.float32)  # 200x200 크기로 변환
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.imread(image)
    marker_crop = cv2.warpPerspective(img, M, (200,200))

    # 마커 부분에 대해 2차 Laplacian 블러 검출
    marker_gray = cv2.cvtColor(marker_crop, cv2.COLOR_BGR2GRAY)
    marker_var = cv2.Laplacian(marker_gray, cv2.CV_64F).var()
    total_var += marker_var
    print(f'Marker Crop Variance: {marker_var:.2f}')
    if marker_var < BLUR_THRESHOLD_2ND:
        print(f'Image {os.path.relpath(image)} marker region is blurred, skipping further processing.\n')
        return False

    R, _ = cv2.Rodrigues(rvecs[0])
    p_y_r = rotation_matrix_to_euler_angles(R)
    p_y_r = np.degrees(p_y_r)
    relative_tvecs, relative_rvecs = convert_marker_relative_6dof_to_camera_relative_6dof(tvecs, p_y_r)

    # pitch, yaw < 20
    if np.abs(relative_rvecs[0]) + np.abs(relative_rvecs[1]) + np.abs(relative_rvecs[2]) > 50:
        print(f'Image {os.path.relpath(image)} has high angle, skipping further processing.\n')
        return False
    
    return True


# In[ ]:


# # 모든 하위 폴더의 jpg 파일까지 탐색
# images = glob.glob("./**/*.jpg", recursive=True)

# total_var = np.array([], dtype=np.float64)
# processed_images = []

# for image in images:
#     var = laplacian_variance(image)
#     print(f'Test Image: {os.path.relpath(image)}')
#     print(f'Variance: {var:.2f}')

#     # 1차 필터링
#     if var < BLUR_THRESHOLD_1ST:
#         print(f'Image {os.path.relpath(image)} is blurred, skipping further processing.')
#         continue
#     else:
#         print(f'Image {os.path.relpath(image)} passed the first blur check.')

#     rvecs, tvecs, _, corners = estimate_aruco_pose(cv2.imread(image), camera_matrix, dist_coeffs, marker_length=0.1)

#     # corners가 없으면 패스
#     if corners is None or len(corners) != 4:
#         print(f'No marker detected in {os.path.relpath(image)}, skipping.\n')
#         continue

#     # 마커 부분만 perspective transform으로 crop
#     src_pts = np.array(corners, dtype=np.float32)
#     dst_pts = np.array([[0,0],[199,0],[199,199],[0,199]], dtype=np.float32)  # 200x200 크기로 변환
#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#     img = cv2.imread(image)
#     marker_crop = cv2.warpPerspective(img, M, (200,200))

#     # 마커 부분에 대해 2차 Laplacian 블러 검출
#     marker_gray = cv2.cvtColor(marker_crop, cv2.COLOR_BGR2GRAY)
#     marker_var = cv2.Laplacian(marker_gray, cv2.CV_64F).var()
#     total_var += marker_var
#     print(f'Marker Crop Variance: {marker_var:.2f}')
#     if marker_var < BLUR_THRESHOLD_2ND:
#         print(f'Image {os.path.relpath(image)} marker region is blurred, skipping further processing.\n')
#         continue

#     R, _ = cv2.Rodrigues(rvecs[0])
#     p_y_r = rotation_matrix_to_euler_angles(R)
#     p_y_r = np.degrees(p_y_r)
#     relative_tvecs, relative_rvecs = convert_marker_relative_6dof_to_camera_relative_6dof(tvecs, p_y_r)

#     # pitch, yaw < 20
#     if np.abs(relative_rvecs[0]) > 20 and np.abs(relative_rvecs[1] > 20):
#         print(f'Image {os.path.relpath(image)} has pitch or yaw > 20, skipping further processing.\n')
#         continue
#     processed_images.append(image)

# print(f'number of processed images: {len(processed_images)}')


# In[10]:


print(f'mean_marker_var: {total_var/len(images)}')
mean = total_var / len(images)


# In[ ]:




