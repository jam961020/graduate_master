"""
BoRisk CVaR Optimization - 완전 수정 버전
핵심 수정사항:
1. GP 정규화 및 안정화
2. 획득함수 개선 (UCB/EI 전환)
3. CVaR 계산 수정
4. 초기 샘플링 후 GP 업데이트
5. 로깅 개선 (점만 추가)
"""
import torch
import numpy as np
import cv2
import json
import sys
import traceback
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

# 전체 파이프라인 import
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# 9D: AirLine 파라미터 (6D) + RANSAC 가중치 (3D)
BOUNDS = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 1],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 10]
], dtype=DTYPE, device=DEVICE)


def line_equation_evaluation(detected_coords, gt_coords, image_size=(640, 480),
                            direction_weight=0.6, distance_weight=0.4):
    """
    직선 방정식 기반 평가: 방향 유사도 + 평행 거리
    끝점이 아닌 직선 자체의 방정식으로 평가

    Args:
        detected_coords: 검출된 좌표 dict
        gt_coords: GT 좌표 dict
        image_size: (width, height)
        direction_weight: 방향 가중치
        distance_weight: 거리 가중치

    Returns:
        score: float [0, 1]
    """
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]

    diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
    distance_threshold = diagonal * 0.05  # 5%

    line_scores = []

    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)

        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)

        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue

        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue

        # 직선 방정식: Ax + By + C = 0
        def to_line_eq(x1, y1, x2, y2):
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            norm = np.sqrt(A**2 + B**2) + 1e-6
            return A/norm, B/norm, C/norm

        A_gt, B_gt, C_gt = to_line_eq(gt_x1, gt_y1, gt_x2, gt_y2)
        A_det, B_det, C_det = to_line_eq(det_x1, det_y1, det_x2, det_y2)

        # 1. 방향 유사도 (법선 벡터 내적)
        direction_sim = abs(A_gt*A_det + B_gt*B_det)  # [0, 1]

        # 2. 평행 거리 (GT 직선의 중점에서 검출 직선까지)
        mid_x = (gt_x1 + gt_x2) / 2
        mid_y = (gt_y1 + gt_y2) / 2
        parallel_dist = abs(A_det*mid_x + B_det*mid_y + C_det)

        distance_sim = max(0.0, 1.0 - (parallel_dist / distance_threshold))

        # 3. 종합 점수
        line_score = direction_weight * direction_sim + distance_weight * distance_sim
        line_scores.append(line_score)

    if len(line_scores) == 0:
        return 0.0

    return float(np.mean(line_scores))


def load_dataset(image_dir, gt_file, complete_only=False, n_augment=0):
    """
    전체 데이터셋 로드
    
    Args:
        complete_only: True면 12개 좌표 모두 있는 이미지만 로드
        n_augment: 0보다 크면 각 이미지를 n_augment개로 증강
    """
    image_dir = Path(image_dir)
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    images_data = []
    
    for img_name, data in gt_data.items():
        possible_paths = [
            image_dir / f"{img_name}.jpg",
            image_dir / f"{img_name}.png",
        ]
        
        img_path = None
        for p in possible_paths:
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        gt_coords = data.get('coordinates', data)
        
        # 완전한 데이터만 필터링
        if complete_only:
            required_keys = [
                'longi_left_lower_x', 'longi_left_lower_y',
                'longi_right_lower_x', 'longi_right_lower_y',
                'longi_left_upper_x', 'longi_left_upper_y',
                'longi_right_upper_x', 'longi_right_upper_y',
                'collar_left_lower_x', 'collar_left_lower_y',
                'collar_left_upper_x', 'collar_left_upper_y'
            ]
            if not all(gt_coords.get(k, 0) != 0 for k in required_keys):
                continue
        
        # 원본 추가
        img_data = {
            'name': img_name,
            'image': image,
            'gt_coords': gt_coords,
            'is_augmented': False
        }
        images_data.append(img_data)
        
        # 증강 이미지 생성
        if n_augment > 0:
            augmented_images = augment_image(image, n_augment)
            for aug_idx, aug_img in enumerate(augmented_images):
                aug_data = {
                    'name': f"{img_name}_aug{aug_idx}",
                    'image': aug_img,
                    'gt_coords': gt_coords,
                    'is_augmented': True
                }
                images_data.append(aug_data)
    
    return images_data


def augment_image(image, n_augment=5):
    """이미지 증강: 노이즈 및 밝기/대비 변화"""
    augmented = []
    
    for i in range(n_augment):
        aug = image.copy().astype(np.float32)
        
        # 1. Gaussian noise
        noise_sigma = np.random.uniform(2, 8)
        noise = np.random.normal(0, noise_sigma, image.shape)
        aug = aug + noise
        
        # 2. Brightness
        brightness = np.random.uniform(0.85, 1.15)
        aug = aug * brightness
        
        # 3. Contrast
        contrast = np.random.uniform(0.9, 1.1)
        aug = (aug - 128) * contrast + 128
        
        aug = np.clip(aug, 0, 255).astype(np.uint8)
        augmented.append(aug)
    
    return augmented


def objective_function(X, images_data, yolo_detector, alpha=0.3, verbose=False):
    """
    CVaR 목적 함수

    Args:
        X: [1, 9] 파라미터 (AirLine 6D + RANSAC 3D)
        images_data: 전체 이미지 데이터
        yolo_detector: YOLO 검출기
        alpha: CVaR의 α (worst α% of cases)
        verbose: 디버깅 출력

    Returns:
        cvar_score: float
    """
    params = {
        'edgeThresh1': X[0, 0].item(),
        'simThresh1': X[0, 1].item(),
        'pixelRatio1': X[0, 2].item(),
        'edgeThresh2': X[0, 3].item(),
        'simThresh2': X[0, 4].item(),
        'pixelRatio2': X[0, 5].item(),
        'ransac_center_w': X[0, 6].item(),
        'ransac_length_w': X[0, 7].item(),
        'ransac_consensus_w': int(X[0, 8].item()),
    }

    scores = []

    for i, img_data in enumerate(images_data):
        try:
            image = img_data['image']
            gt_coords = img_data['gt_coords']

            detected_coords = detect_with_full_pipeline(image, params, yolo_detector)

            h, w = image.shape[:2]
            score = line_equation_evaluation(detected_coords, gt_coords, image_size=(w, h))
            scores.append(score)

        except KeyboardInterrupt:
            raise
        except Exception:
            scores.append(0.0)

    scores = np.array(scores)

    if len(scores) == 0:
        return 0.0

    n_worst = max(1, int(len(scores) * alpha))
    worst_scores = np.sort(scores)[:n_worst]
    cvar = np.mean(worst_scores)

    if verbose:
        print(f"CVaR={cvar:.4f} (mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f})")

    return cvar


def optimize_risk_aware_bo(images_data, yolo_detector, metric="lp", 
                           n_iterations=30, n_initial=15, alpha=0.3):
    """
    메인 최적화 루프 (완전 수정됨)
    
    주요 개선사항:
    1. 초기 샘플링 후 즉시 GP 학습
    2. Y 값 정규화
    3. 탐험-활용 밸런싱 (UCB → EI)
    4. 획득함수 최적화 개선
    """
    print("\n" + "="*60)
    print(f"Risk-aware BO with CVaR")
    print(f"Total images: {len(images_data)}")
    print(f"CVaR α: {alpha} (worst {int(alpha*100)}%)")
    print("="*60)
    
    # 1. 초기 샘플링
    print(f"\n[Phase 1] Initial sampling ({n_initial} samples)")

    sobol = SobolEngine(dimension=9, scramble=True)
    X_init = sobol.draw(n_initial).to(dtype=DTYPE, device=DEVICE)
    X_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * X_init
    
    Y_init = []
    for i, x in enumerate(X_init):
        score = objective_function(x.unsqueeze(0), images_data, yolo_detector,
                                  alpha=alpha, verbose=False)
        Y_init.append(score)
        print(f"Init {i+1}/{n_initial}: CVaR={score:.4f}")
    
    Y_init = torch.tensor(Y_init, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    
    # 2. Y 값 정규화 (중요!)
    Y_mean = Y_init.mean()
    Y_std = Y_init.std()
    
    if Y_std < 1e-6:
        print("\n⚠️ Warning: Low variance in Y values, adding noise...")
        Y_init = Y_init + 0.01 * torch.randn_like(Y_init)
        Y_mean = Y_init.mean()
        Y_std = Y_init.std()
    
    Y_normalized = (Y_init - Y_mean) / Y_std
    
    print(f"Init complete: mean={Y_mean:.4f}, std={Y_std:.4f}, best={Y_init.max().item():.4f}")

    # 3. 초기 GP 학습 (중요!)
    X = X_init
    Y = Y_normalized
    Y_original = Y_init

    print(f"\n[Phase 2] GP training")
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    print(f"GP noise level: {gp.likelihood.noise.item():.6f}")
    
    best_observed = [Y_original.max().item()]

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 4. BO 루프
    print(f"\n[Phase 3] BO iterations")
    print("-"*60)
    
    for iteration in range(n_iterations):
        # 4.1 탐험-활용 전략
        if iteration < n_iterations // 3:
            # 초반: 탐험 (UCB)
            beta = 2.0
            acq_func = qUpperConfidenceBound(gp, beta=beta)
            acq_name = "UCB"
        else:
            # 후반: 활용 (EI)
            acq_func = qExpectedImprovement(gp, Y.max())
            acq_name = "EI"
        
        # 4.2 획득함수 최적화
        try:
            candidate, acq_value = optimize_acqf(
                acq_func,
                bounds=BOUNDS,
                q=1,
                num_restarts=20,  # 증가
                raw_samples=1024  # 증가
            )
        except Exception as e:
            print(f"\n⚠️ Acquisition optimization failed: {e}")
            # 폴백: 랜덤 샘플링
            candidate = torch.rand(1, 9, dtype=DTYPE, device=DEVICE)
            candidate = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * candidate
            acq_value = torch.tensor(0.0)
        
        # 4.3 평가 (단 1회만 실행됨)
        new_y = objective_function(candidate, images_data, yolo_detector,
                                  alpha=alpha, verbose=False)

        # 정규화
        new_y_normalized = (new_y - Y_mean.item()) / (Y_std.item() + 1e-6)

        # 4.3.1 반복마다 로그 저장
        iter_log = {
            "iteration": iteration + 1,
            "acq_function": acq_name,
            "acq_value": float(acq_value.item()),
            "parameters": {
                "edgeThresh1": float(candidate[0, 0].item()),
                "simThresh1": float(candidate[0, 1].item()),
                "pixelRatio1": float(candidate[0, 2].item()),
                "edgeThresh2": float(candidate[0, 3].item()),
                "simThresh2": float(candidate[0, 4].item()),
                "pixelRatio2": float(candidate[0, 5].item()),
                "ransac_center_w": float(candidate[0, 6].item()),
                "ransac_length_w": float(candidate[0, 7].item()),
                "ransac_consensus_w": int(candidate[0, 8].item()),
            },
            "cvar": float(new_y),
            "cvar_normalized": float(new_y_normalized)
        }

        log_file = log_dir / f"iter_{iteration+1:03d}.json"
        with open(log_file, 'w') as f:
            json.dump(iter_log, f, indent=2)
        
        # 4.4 데이터 추가
        X = torch.cat([X, candidate])
        Y = torch.cat([Y, torch.tensor([[new_y_normalized]], dtype=DTYPE, device=DEVICE)])
        Y_original = torch.cat([Y_original, torch.tensor([[new_y]], dtype=DTYPE, device=DEVICE)])
        
        # 4.5 GP 재학습
        try:
            gp = SingleTaskGP(X, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  ⚠️ GP refit failed: {e}")
            # 노이즈 추가 후 재시도
            Y = Y + 0.01 * torch.randn_like(Y)
            gp = SingleTaskGP(X, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        
        best_observed.append(Y_original.max().item())
        improvement = best_observed[-1] - best_observed[-2]

        print(f"Iter {iteration+1}/{n_iterations} ({acq_name}): CVaR={new_y:.4f}, Best={best_observed[-1]:.4f} ({improvement:+.4f})")
        
        # 4.6 조기 종료 체크
        if iteration > 10:
            recent_improvements = [best_observed[i] - best_observed[i-1] 
                                  for i in range(-5, 0)]
            if all(abs(imp) < 1e-5 for imp in recent_improvements):
                print("\n조기 종료: 최근 5회 개선 없음")
                break
    
    # 5. 최종 결과
    best_idx = Y_original.argmax()
    best_X = X[best_idx]
    
    print("\n" + "="*60)
    print("Optimization Complete")
    print("="*60)
    
    return best_X, best_observed, Y_original


if __name__ == "__main__":
    import sys
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description="Risk-aware BO for Welding")
    parser.add_argument("--image_dir", default="dataset/images/test", help="Image directory")
    parser.add_argument("--gt_file", default="dataset/ground_truth.json", help="Ground truth JSON")
    parser.add_argument("--yolo_model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--iterations", type=int, default=20, help="BO iterations")
    parser.add_argument("--n_initial", type=int, default=15, help="Initial samples")
    parser.add_argument("--alpha", type=float, default=0.3, help="CVaR alpha (worst percent)")
    parser.add_argument("--complete_only", action="store_true", help="Use only complete GT images")
    parser.add_argument("--n_augment", type=int, default=0, help="Number of augmentations per image")
    args = parser.parse_args()
    
    # GT 파일 확인
    gt_file = Path(args.gt_file)
    if not gt_file.exists():
        print(f"[ERROR] GT 파일 없음: {args.gt_file}")
        print("\n먼저 라벨링을 완료하세요:")
        print("  python labeling_tool.py")
        sys.exit(1)
    
    # YOLO 모델 확인
    yolo_model_path = Path(args.yolo_model)
    if not yolo_model_path.exists():
        print(f"[ERROR] YOLO 모델 없음: {args.yolo_model}")
        sys.exit(1)
    
    # 데이터셋 로드
    print(f"Loading dataset from {args.image_dir}...")
    images_data = load_dataset(args.image_dir, gt_file, 
                              complete_only=args.complete_only,
                              n_augment=args.n_augment)
    
    if len(images_data) == 0:
        print("[ERROR] 로드된 이미지가 없습니다!")
        sys.exit(1)
    
    # 원본과 증강 이미지 개수 출력
    n_original = sum(1 for img in images_data if not img.get('is_augmented', False))
    n_augmented = len(images_data) - n_original
    
    print(f"Loaded {n_original} original images")
    if n_augmented > 0:
        print(f"Generated {n_augmented} augmented images (total: {len(images_data)})")
    
    # YOLO 검출기 초기화
    print(f"\nInitializing YOLO detector...")
    yolo_detector = YOLODetector(args.yolo_model)
    
    # 최적화 실행
    best_params, history, all_Y = optimize_risk_aware_bo(
        images_data,
        yolo_detector,
        n_iterations=args.iterations,
        n_initial=args.n_initial,
        alpha=args.alpha
    )
    
    # 결과 출력
    print("\n최적 파라미터:")
    print(f"  edgeThresh1:        {best_params[0]:7.2f}")
    print(f"  simThresh1:         {best_params[1]:7.4f}")
    print(f"  pixelRatio1:        {best_params[2]:7.4f}")
    print(f"  edgeThresh2:        {best_params[3]:7.2f}")
    print(f"  simThresh2:         {best_params[4]:7.4f}")
    print(f"  pixelRatio2:        {best_params[5]:7.4f}")
    print(f"  ransac_center_w:    {best_params[6]:7.4f}")
    print(f"  ransac_length_w:    {best_params[7]:7.4f}")
    print(f"  ransac_consensus_w: {int(best_params[8]):7d}")
    
    print(f"\n성능:")
    print(f"  최종 CVaR: {history[-1]:.4f}")
    print(f"  초기 CVaR: {history[0]:.4f}")
    print(f"  개선도: {(history[-1] - history[0]) / (history[0] + 1e-6) * 100:+.1f}%")
    
    # 결과 저장
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"bo_cvar_{timestamp}.json"
    
    result_data = {
        "alpha": args.alpha,
        "n_images": len(images_data),
        "n_original": n_original,
        "n_augmented": n_augmented,
        "iterations": args.iterations,
        "best_params": {
            "edgeThresh1": float(best_params[0]),
            "simThresh1": float(best_params[1]),
            "pixelRatio1": float(best_params[2]),
            "edgeThresh2": float(best_params[3]),
            "simThresh2": float(best_params[4]),
            "pixelRatio2": float(best_params[5]),
            "ransac_center_w": float(best_params[6]),
            "ransac_length_w": float(best_params[7]),
            "ransac_consensus_w": int(best_params[8]),
        },
        "history": [float(x) for x in history],
        "final_cvar": float(history[-1]),
        "improvement": float((history[-1] - history[0]) / (history[0] + 1e-6) * 100),
        "complete_only": args.complete_only,
        "n_augment": args.n_augment
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n결과 저장: {result_file}")
    print("="*60)