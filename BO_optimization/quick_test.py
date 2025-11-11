#!/usr/bin/env python
"""Quick test to check if optimization.py imports work"""
import sys
print("Starting quick test...")

try:
    print("1. Importing torch...")
    import torch
    print(f"   ✓ torch {torch.__version__}")

    print("2. Importing botorch...")
    from botorch.models import SingleTaskGP
    print("   ✓ botorch OK")

    print("3. Importing full_pipeline...")
    from full_pipeline import detect_with_full_pipeline
    print("   ✓ full_pipeline OK")

    print("4. Importing yolo_detector...")
    from yolo_detector import YOLODetector
    print("   ✓ yolo_detector OK")

    print("5. Importing environment_independent...")
    from environment_independent import extract_parameter_independent_environment
    print("   ✓ environment_independent OK")

    print("6. Checking dataset...")
    from pathlib import Path
    gt_file = Path("dataset/ground_truth.json")
    if gt_file.exists():
        print(f"   ✓ GT file exists: {gt_file}")
    else:
        print(f"   ✗ GT file missing: {gt_file}")

    yolo_model = Path("models/best.pt")
    if yolo_model.exists():
        print(f"   ✓ YOLO model exists: {yolo_model}")
    else:
        print(f"   ✗ YOLO model missing: {yolo_model}")

    print("\n✅ All imports successful!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
