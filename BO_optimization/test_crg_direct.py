"""
CRG311.desGrow() 직접 호출 테스트
"""
import sys
import numpy as np
import torch
import cv2

print("="*60)
print("CRG311.desGrow() DIRECT TEST")
print("="*60)

print("\nStep 1: Import CRG311...")
sys.path.insert(0, '/home/jeongho/projects/graduate/YOLO_AirLine')
import CRG311
print("  ✓ SUCCESS")

print("\nStep 2: Prepare test data...")
# 작은 테스트 이미지 생성
h, w = 480, 640
test_edges = np.zeros((1, h, w), dtype=np.uint8)  # 3D: [1, H, W]
# 몇 개 엣지 추가
test_edges[0, 200:210, 100:500] = 1
test_edges[0, 300:310, 150:450] = 1

# ODes: Orientation Descriptors (가상 데이터)
ODes_np = np.random.rand(10, h, w).astype(np.float32)

# Used map
used_map = np.ones((h, w), dtype=np.int32)

# Output arrays
out = np.zeros((1, 1000, 2), dtype=np.float32)  # 3D: [1, N, 2]?

# TMP arrays
TMP1 = np.zeros((50000, 2), dtype=np.int32)
TMP2 = np.zeros((2, 300000, 2), dtype=np.int32)
TMP3 = np.zeros((3000, 2, 2), dtype=np.float32)

THETA_RES = 10

# 파라미터
simThresh = 0.9
pixelNumThresh = int(np.hypot(h, w) * 0.03)  # 3%

print(f"  Image size: {h}x{w}")
print(f"  pixelNumThresh: {pixelNumThresh}")
print(f"  simThresh: {simThresh}")

print("\nStep 3: Ensure C-contiguity...")
ODes_np = np.ascontiguousarray(ODes_np, dtype=np.float32)
test_edges = np.ascontiguousarray(test_edges, dtype=np.uint8)
used_map = np.ascontiguousarray(used_map, dtype=np.int32)
out = np.ascontiguousarray(out, dtype=np.float32)
TMP1 = np.ascontiguousarray(TMP1, dtype=np.int32)
TMP2 = np.ascontiguousarray(TMP2, dtype=np.int32)
TMP3 = np.ascontiguousarray(TMP3, dtype=np.float32)
print("  ✓ All arrays contiguous")

print("\nStep 4: Calling CRG311.desGrow()...")
try:
    rawLineNum = CRG311.desGrow(
        test_edges,
        used_map,
        ODes_np,
        out,
        simThresh,
        pixelNumThresh,
        TMP1,
        TMP2,
        TMP3,
        THETA_RES
    )
    print(f"  ✓ SUCCESS: Detected {rawLineNum} lines")
    print("\n" + "="*60)
    print("CRG311.desGrow() WORKS!")
    print("="*60)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
