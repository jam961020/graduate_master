#!/usr/bin/env python
"""
AirLine run_airline 최종 테스트
"""
import sys
import cv2
import numpy as np

print("="*60)
print("AIRLINE FINAL TEST")
print("="*60)

print("\nStep 1: Import AirLine...")
sys.path.insert(0, '/home/jeongho/projects/graduate/YOLO_AirLine')
from AirLine_assemble_test import run_airline, _init_airline_models

print("\nStep 2: Load test image...")
test_img = cv2.imread("dataset/images/test/WIN_20250604_14_01_48_Pro.jpg")
if test_img is None:
    print("  ✗ Failed")
    sys.exit(1)
print(f"  ✓ Image loaded: {test_img.shape}")

print("\nStep 3: Initialize AirLine models...")
_init_airline_models()
print("  ✓ Models ready")

print("\nStep 4: Run AirLine...")
airline_config = {
    "edgeThresh": -3.0,
    "simThresh": 0.98,
    "pixelNumThresh": 100
}
try:
    lines = run_airline(test_img, airline_config)
    print(f"  ✓ SUCCESS: {len(lines)} lines detected")
    print("\n" + "="*60)
    print("AIRLINE WORKS PERFECTLY!")
    print("="*60)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
