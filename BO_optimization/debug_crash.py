"""
Segfault 디버깅 스크립트 - 단계별 테스트
"""
import sys
import torch
from pathlib import Path

print("="*60)
print("SEGFAULT DEBUG - 단계별 테스트")
print("="*60)

# Step 1: 기본 import
print("\n[1/7] Import optimization module...")
try:
    import optimization
    print("✓ SUCCESS")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Step 2: 데이터 로드
print("\n[2/7] Load dataset...")
try:
    import json
    dataset_path = Path("dataset/images/test")
    gt_path = Path("dataset/ground_truth.json")

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    images_data = []
    for img_name, gt_data in list(ground_truth.items())[:3]:  # 3개만 테스트
        img_path = dataset_path / f"{img_name}.jpg"
        if img_path.exists():
            images_data.append({
                'name': img_name,
                'path': str(img_path),
                'gt': gt_data
            })

    print(f"✓ SUCCESS: Loaded {len(images_data)} images")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: YOLO 초기화
print("\n[3/7] Initialize YOLO detector...")
try:
    from yolo_detector import YOLODetector
    yolo = YOLODetector("models/best.pt")
    print("✓ SUCCESS")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 환경 벡터 추출
print("\n[4/7] Extract environment vector...")
try:
    from environment_independent import extract_parameter_independent_environment
    import cv2

    test_img = cv2.imread(images_data[0]['path'])
    env = extract_parameter_independent_environment(test_img)
    print(f"✓ SUCCESS: {env}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Full pipeline 테스트 (가장 의심스러운 부분)
print("\n[5/7] Test full pipeline (AirLine)...")
try:
    from full_pipeline import detect_with_full_pipeline

    # 기본 파라미터로 테스트
    test_params = {
        'edgeThresh1': -3.0,
        'simThresh1': 0.98,
        'pixelRatio1': 0.05,
        'edgeThresh2': 1.0,
        'simThresh2': 0.75,
        'pixelRatio2': 0.05,
        'ransac_center_w': 0.5,
        'ransac_length_w': 0.5,
        'ransac_consensus_w': 5
    }

    print("  Calling detect_with_full_pipeline...")
    result = detect_with_full_pipeline(
        test_img,
        test_params,
        yolo
    )
    print(f"✓ SUCCESS: {result}")
except Exception as e:
    print(f"✗ FAILED (THIS IS LIKELY THE CRASH POINT): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Objective function 테스트
print("\n[6/7] Test objective function...")
try:
    X = torch.tensor([[-3.0, 0.98, 0.05, 1.0, 0.75, 0.05, 0.5, 0.5, 5]],
                     dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    score = optimization.objective_function(X, images_data, yolo, alpha=0.3, verbose=False)
    print(f"✓ SUCCESS: CVaR = {score}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: 초기화 Sobol 샘플링
print("\n[7/7] Test Sobol initialization...")
try:
    from torch.quasirandom import SobolEngine
    sobol = SobolEngine(dimension=9, scramble=True)
    X_init = sobol.draw(n=2).to(dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    BOUNDS = torch.tensor([
        [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 1],
        [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 10]
    ], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    X_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * X_init
    print(f"✓ SUCCESS: Generated {X_init.shape[0]} samples")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! The crash is elsewhere.")
print("="*60)
