"""
DexiNed 모델 테스트
"""
import sys
import torch
import numpy as np
import cv2

print("Step 1: Import DexiNed...")
sys.path.insert(0, '/home/jeongho/projects/graduate/YOLO_AirLine')
from deximodel import DexiNed

print("Step 2: Create model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
model = DexiNed().to(device)

print("Step 3: Load weights...")
import os
dexi_path = "/home/jeongho/projects/graduate/YOLO_AirLine/dexi.pth"
if os.path.exists(dexi_path):
    state_dict = torch.load(dexi_path, map_location=device)
    model.load_state_dict(state_dict)
    print("  ✓ Loaded")
else:
    print(f"  ✗ File not found: {dexi_path}")

print("Step 4: Test inference...")
# 작은 더미 이미지 생성
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_tensor = torch.from_numpy(test_img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

print("  Running forward pass...")
with torch.no_grad():
    output = model(test_tensor)

print(f"✓ SUCCESS: Output shape = {output[0].shape}")
print("DexiNed is working!")
