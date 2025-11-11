"""
전체 실험 자동 실행
"""
import cv2
import json
import datetime
from pathlib import Path
from optimization import optimize_risk_aware_bo


def main():
    # 설정
    IMAGE_DIR = Path("../data/images")
    GT_FILE = Path("../data/ground_truth.json")
    RESULTS_DIR = Path("../results")
    
    # GT 로드
    if not GT_FILE.exists():
        print(f"[ERROR] Ground truth not found: {GT_FILE}")
        print("Please run labeling_tool.py first!")
        return
    
    with open(GT_FILE, 'r') as f:
        gt_data = json.load(f)
    
    labeled_images = list(gt_data.keys())
    print(f"Found {len(labeled_images)} labeled images")
    
    if len(labeled_images) == 0:
        print("No labeled images! Run labeling_tool.py first.")
        return
    
    # 실험 폴더 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / f"experiment_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults will be saved to: {exp_dir}\n")
    
    # 두 메트릭으로 실험
    metrics = ["lp", "endpoint"]
    
    all_results = {}
    
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"METRIC: {metric.upper()}")
        print(f"{'='*60}")
        
        metric_results = {}
        
        for img_name in labeled_images:
            print(f"\n[{img_name}] Processing...")
            
            # 이미지 로드
            img_path = IMAGE_DIR / f"{img_name}.jpg"
            if not img_path.exists():
                img_path = IMAGE_DIR / f"{img_name}.png"
            
            if not img_path.exists():
                print(f"  [SKIP] Image not found: {img_name}")
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [SKIP] Failed to load: {img_name}")
                continue
            
            # 최적화 실행
            try:
                best_params, history, all_Y = optimize_risk_aware_bo(
                    image,
                    img_name,
                    metric=metric,
                    n_iterations=20,  # 빠른 테스트용
                    n_initial=5,
                    n_w=8
                )
                
                metric_results[img_name] = {
                    "best_params": {
                        "edgeThresh1": float(best_params[0]),
                        "simThresh1": float(best_params[1]),
                        "pixelRatio1": float(best_params[2]),
                        "edgeThresh2": float(best_params[3]),
                        "simThresh2": float(best_params[4]),
                        "pixelRatio2": float(best_params[5]),
                    },
                    "history": [float(x) for x in history],
                    "final_cvar": float(history[-1]),
                    "improvement": float((history[-1] - history[0]) / (history[0] + 1e-6) * 100)
                }
                
                print(f"  [OK] Final CVaR: {history[-1]:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue
        
        all_results[metric] = metric_results
        
        # 메트릭별 결과 저장
        metric_file = exp_dir / f"{metric}_results.json"
        with open(metric_file, 'w') as f:
            json.dump(metric_results, f, indent=2)
        
        print(f"\n{metric.upper()} results saved to: {metric_file}")
    
    # 전체 결과 저장
    summary_file = exp_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()