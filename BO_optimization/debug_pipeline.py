"""
파이프라인 디버그 스크립트
CVaR이 0으로 나오는 원인 파악
"""
import torch
import numpy as np
from pathlib import Path

from optimization import load_dataset, objective_function
from yolo_detector import YOLODetector
from evaluation import GT_LABELS


def main():
    print("="*60)
    print("Pipeline Debug Script")
    print("="*60)
    
    # 1. 데이터셋 로드
    print("\n[1] Loading dataset...")
    try:
        # [수정] 증강 포함하여 로드
        images_data = load_dataset('dataset/images/test', 'dataset/ground_truth.json', 
                                   complete_only=False, n_augment=5)
        
        n_original = sum(1 for img in images_data if not img.get('is_augmented', False))
        n_augmented = sum(1 for img in images_data if img.get('is_augmented', False))
        
        print(f"✓ Loaded {n_original} original images")
        if n_augmented > 0:
            print(f"✓ Generated {n_augmented} augmented images")
            print(f"✓ Total: {len(images_data)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # 이미지 이름 출력
    print(f"\nFirst 5 sample names:")
    for i, img in enumerate(images_data[:5]):
        aug_marker = " [AUG]" if img.get('is_augmented', False) else ""
        print(f"  {i+1}. {img['name']}{aug_marker}")
    
    # 2. Ground Truth 확인
    print(f"\n[2] Checking Ground Truth...")
    print(f"GT has {len(GT_LABELS)} entries")
    print(f"\nFirst 5 GT keys:")
    for i, key in enumerate(list(GT_LABELS.keys())[:5]):
        print(f"  {i+1}. {key}")
    
    # 3. 이름 매칭 확인
    print(f"\n[3] Checking name matching...")
    matched = 0
    unmatched = []
    for img in images_data:
        if img['name'] in GT_LABELS:
            matched += 1
        else:
            unmatched.append(img['name'])
    
    print(f"Matched: {matched}/{len(images_data)}")
    if unmatched:
        print(f"Unmatched images:")
        for name in unmatched[:5]:
            print(f"  - {name}")
    
    if matched == 0:
        print("\n✗ ERROR: No images matched with GT!")
        print("\nPossible issues:")
        print("  1. Image filenames don't match GT keys")
        print("  2. GT file has different naming convention")
        print("  3. Images in wrong directory")
        return
    
    # 4. YOLO 모델 로드
    print(f"\n[4] Loading YOLO model...")
    try:
        yolo = YOLODetector('models/best.pt')
        print("✓ YOLO model loaded")
    except Exception as e:
        print(f"✗ Failed to load YOLO: {e}")
        return
    
    # 5. 첫 번째 매칭된 이미지로 테스트
    print(f"\n[5] Testing pipeline on first matched image...")
    
    # 매칭된 이미지 찾기
    test_img_data = None
    for img in images_data:
        if img['name'] in GT_LABELS:
            test_img_data = img
            break
    
    if not test_img_data:
        print("✗ No matched image found for testing")
        return
    
    print(f"Testing with: {test_img_data['name']}")
    
    # 테스트 파라미터
    test_params = {
        'edgeThresh1': -3.0,
        'simThresh1': 0.98,
        'pixelRatio1': 0.05,
        'edgeThresh2': 1.0,
        'simThresh2': 0.75,
        'pixelRatio2': 0.05
    }
    
    # Tensor로 변환
    X = torch.tensor([[
        test_params['edgeThresh1'],
        test_params['simThresh1'],
        test_params['pixelRatio1'],
        test_params['edgeThresh2'],
        test_params['simThresh2'],
        test_params['pixelRatio2']
    ]], dtype=torch.double)
    
    print("\nRunning detection...")
    print("-" * 60)
    
    try:
        score = objective_function(
            X,
            [test_img_data],
            yolo,
            metric='lp',
            verbose=True
        )
        
        print("-" * 60)
        print(f"\n✓ Detection complete!")
        print(f"Score: {score:.4f}")
        
        if score == 0.0:
            print("\n⚠ WARNING: Score is 0.0")
            print("Possible reasons:")
            print("  1. No coordinates detected")
            print("  2. GT data is missing or invalid")
            print("  3. Detection failed completely")
            print("  4. Evaluation metric issue")
        else:
            print(f"\n✓ SUCCESS: Score is non-zero ({score:.4f})")
            
    except Exception as e:
        print(f"\n✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 전체 이미지 통계
    print(f"\n[6] Running on ALL samples (original + augmented)...")
    print("This will take a while...")
    
    matched_images = [img for img in images_data if img['name'].split('_aug')[0] in GT_LABELS or img['name'] in GT_LABELS]
    print(f"Testing {len(matched_images)} samples...")
    
    try:
        all_scores = objective_function(
            X,
            matched_images,
            yolo,
            metric='lp',
            verbose=False
        )
        
        print(f"\n✓ CVaR (α=0.1): {all_scores:.4f}")
        
        if all_scores > 0:
            print(f"✓ SUCCESS: CVaR is non-zero!")
        else:
            print(f"⚠ WARNING: CVaR is still 0.0")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Debug Complete")
    print("="*60)


if __name__ == "__main__":
    main()