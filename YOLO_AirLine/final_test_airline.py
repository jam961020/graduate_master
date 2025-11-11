"""
원본 Coordinate Detector.py의 전역 변수들을 그대로 사용
"""
import os
import cv2
import numpy as np
import torch
import sys
import pathlib
from ultralytics import YOLO

# CRG311 import
AIRLINE_DIR = pathlib.Path(r"C:\Users\user\Desktop\study\task\weld2025\AirLine\build")
sys.path.append(str(AIRLINE_DIR))
import CRG311 as crg

from deximodel import DexiNed
import torch.nn as nn

THETARESOLUTION = 6
KERNEL_SIZE = 9

def buildOrientationDetector():
    thetaN = nn.Conv2d(1, THETARESOLUTION, KERNEL_SIZE, 1, KERNEL_SIZE // 2, bias=False).cuda()
    for i in range(THETARESOLUTION):
        kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
        angle = i * 180 / THETARESOLUTION
        x = (np.cos(angle / 180 * np.pi) * (KERNEL_SIZE // 2)).astype(np.int32)
        y = (np.sin(angle / 180 * np.pi) * (KERNEL_SIZE // 2)).astype(np.int32)
        
        cv2.line(kernel, (KERNEL_SIZE // 2 - x, KERNEL_SIZE // 2 - y), 
                 (KERNEL_SIZE // 2 + x, KERNEL_SIZE // 2 + y), 1, 1)
        thetaN.weight.data[i] = torch.tensor(kernel, dtype=torch.float32)
    return thetaN

# 원본과 동일한 전역 초기화
print("전역 변수 초기화 중...")
model = YOLO(r".\best_l.pt")
OrientationDetector = buildOrientationDetector()
edgeDetector = DexiNed().cuda()
edge_state_dict = torch.load(r".\dexi.pth", map_location='cuda:0')
edgeDetector.load_state_dict(edge_state_dict)
edgeDetector.eval()

usingUnet = 0
config = {
    "edgeThresh": 0,
    "simThresh": 0.9,
    "pixelNumThresh": 100,
}

tempMem = np.zeros((50000, 2), dtype=np.int32)
tempMem2 = np.zeros((2, 300000, 2), dtype=np.int32)
tempMem3 = np.zeros((3000, 2, 2), dtype=np.float32)

print("전역 변수 초기화 완료!")

def test_with_global_vars(img):
    """
    원본과 동일한 전역 변수를 사용해서 테스트
    """
    print(f"입력 이미지 크기: {img.shape}")
    
    # 먼저 YOLO 실행 (원본과 동일하게)
    print("YOLO 실행 중...")
    result = model(img, conf=0.3, iou=0.3)
    boxes = result[0].boxes
    print(f"YOLO 박스 개수: {len(boxes)}")
    
    # AirLine 처리
    rx1 = img.copy()
    original_height, original_width = rx1.shape[:2]

    if not usingUnet:
        res = 16
        dscale = 1
        resized_height = rx1.shape[0] // dscale // res * res
        resized_width = rx1.shape[1] // dscale // res * res
        rx1_resized = cv2.resize(rx1, (resized_width, resized_height))
    else:
        rx1_resized = rx1

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    if len(rx1_resized.shape) == 2:
        rx1_resized = cv2.cvtColor(rx1_resized, cv2.COLOR_GRAY2RGB)
    elif rx1_resized.shape[2] == 4:
        rx1_resized = cv2.cvtColor(rx1_resized, cv2.COLOR_RGBA2RGB)

    rx1_resized = np.ascontiguousarray(rx1_resized)
    x1 = rx1_resized

    if usingUnet:
        x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2GRAY)

    x1 = torch.tensor(x1, dtype=torch.float32).cuda() / 255.0

    if usingUnet:
        x1 = x1.unsqueeze(0)
    else:
        x1 = x1.permute(2, 0, 1)

    with torch.no_grad():
        edgeDetection = edgeDetector(x1.unsqueeze(0))
    ODes = OrientationDetector(edgeDetection)
    ODes = torch.nn.functional.normalize(ODes - ODes.mean(1, keepdim=True), p=2.0, dim=1)

    edgeNp = edgeDetection.detach().cpu().numpy()[0, 0]
    print(f"[GLOBAL] Edge 값 범위: {edgeNp.min():.6f} ~ {edgeNp.max():.6f}")
    
    outMap = np.zeros_like(edgeNp, dtype=np.uint8)
    outMap = np.expand_dims(outMap, 2).repeat(3, axis=2)
    out = np.zeros((3000, 2, 3), dtype=np.float32)

    edgeNp = (edgeNp > config["edgeThresh"]).astype(np.uint8) * 255
    print(f"[GLOBAL] Binary edge 픽셀 수: {np.sum(edgeNp > 0)}")

    rawLineNum = crg.desGrow(
        outMap, edgeNp, ODes[0].detach().cpu().numpy(), out,
        config["simThresh"], config["pixelNumThresh"],
        tempMem, tempMem2, tempMem3, THETARESOLUTION
    )
    
    print(f"[GLOBAL] 검출된 선 개수: {rawLineNum}")
    
    return rawLineNum

if __name__ == "__main__":
    img = cv2.imread("../realDataset_1/1_1/WIN_20250604_14_05_51_Pro.jpg")
    result = test_with_global_vars(img)
    print(f"최종 결과: {result} lines") 