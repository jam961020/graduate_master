"""
BoRisk CVaR Optimization - BoRisk 알고리즘 완전 구현
핵심 수정사항:
1. w_set 샘플링 시스템 (n_w개만 평가)
2. GP 모델: (x, w) → y 학습
3. qMultiFidelityKnowledgeGradient 획득 함수
4. CVaR objective 통합
5. 판타지 관측 구조
"""
import torch
import numpy as np
import cv2
import json
import sys
import traceback
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import AppendFeatures
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import (qExpectedImprovement, qUpperConfidenceBound,
                                 qMultiFidelityKnowledgeGradient)
from botorch.acquisition.objective import GenericMCObjective
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

# 전체 파이프라인 import
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector
from environment_independent import extract_parameter_independent_environment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# 8D: AirLine 파라미터 (6D) + RANSAC 가중치 (2D: Q, QG)
BOUNDS = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 1.0, 1.0],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 20.0, 20.0]
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


def extract_all_environments(images_data):
    """
    모든 이미지의 환경 벡터를 사전 추출

    Args:
        images_data: 이미지 데이터 리스트

    Returns:
        env_features: [N, 6] 환경 벡터 텐서
    """
    print("[BoRisk] Extracting environment features from all images...")
    all_env = []

    for img_data in images_data:
        image = img_data['image']
        roi = img_data.get('roi', None)

        # 환경 벡터 추출
        env = extract_parameter_independent_environment(image, roi)
        env_vec = [
            env['brightness'],
            env['contrast'],
            env['edge_density'],
            env['texture_complexity'],
            env['blur_level'],
            env['noise_level']
        ]
        all_env.append(env_vec)

    env_features = torch.tensor(all_env, dtype=DTYPE, device=DEVICE)
    print(f"[BoRisk] Environment features shape: {env_features.shape}")

    return env_features


def sample_w_set(env_features, n_w=15, seed=None):
    """
    w_set 샘플링 (BoRisk의 핵심)

    Args:
        env_features: [N, 6] 전체 환경 벡터
        n_w: 샘플링할 환경 개수
        seed: 랜덤 시드

    Returns:
        w_set: [n_w, 6] 샘플링된 환경 벡터
        w_indices: [n_w] 샘플링된 인덱스
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = env_features.shape[0]
    n_w = min(n_w, N)  # N보다 클 수 없음

    # 랜덤 샘플링
    perm = torch.randperm(N)
    w_indices = perm[:n_w]
    w_set = env_features[w_indices]

    return w_set, w_indices


def evaluate_on_w_set(X, images_data, yolo_detector, w_indices):
    """
    w_set에 해당하는 이미지만 평가 (BoRisk의 핵심!)

    Args:
        X: [1, 8] 파라미터
        images_data: 전체 이미지 데이터
        yolo_detector: YOLO 검출기
        w_indices: [n_w] 평가할 이미지 인덱스

    Returns:
        scores: [n_w, 1] 각 환경에서의 성능
    """
    params = {
        'edgeThresh1': X[0, 0].item(),
        'simThresh1': X[0, 1].item(),
        'pixelRatio1': X[0, 2].item(),
        'edgeThresh2': X[0, 3].item(),
        'simThresh2': X[0, 4].item(),
        'pixelRatio2': X[0, 5].item(),
    }

    # RANSAC 가중치 (Q, QG 개별)
    ransac_weights = (X[0, 6].item(), X[0, 7].item())

    scores = []
    print(f"[DEBUG] Evaluating {len(w_indices)} images with RANSAC weights (Q={ransac_weights[0]:.1f}, QG={ransac_weights[1]:.1f})...")

    for i, idx in enumerate(w_indices):
        try:
            print(f"  [{i+1}/{len(w_indices)}] Image idx={idx.item() if torch.is_tensor(idx) else idx}...", end=' ')
            img_data = images_data[idx.item() if torch.is_tensor(idx) else idx]
            image = img_data['image']
            gt_coords = img_data['gt_coords']

            detected_coords = detect_with_full_pipeline(image, params, yolo_detector, ransac_weights)

            h, w = image.shape[:2]
            score = line_equation_evaluation(detected_coords, gt_coords, image_size=(w, h))
            scores.append(score)
            print(f"score={score:.4f}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            scores.append(0.0)

    return torch.tensor(scores, dtype=DTYPE, device=DEVICE).unsqueeze(-1)


def compute_cvar_from_scores(scores, alpha=0.3):
    """
    스코어 벡터로부터 CVaR 계산

    Args:
        scores: [n] 또는 [n, 1] 스코어 텐서
        alpha: CVaR threshold

    Returns:
        cvar: float
    """
    if scores.dim() > 1:
        scores = scores.squeeze()

    n = scores.shape[0]
    n_worst = max(1, int(n * alpha))

    worst_scores, _ = torch.topk(scores, n_worst, largest=False)
    return worst_scores.mean().item()


def cvar_objective(samples, alpha=0.3):
    """
    CVaR objective for BoRisk

    Args:
        samples: [n_samples, n_w, 1] GP 샘플
        alpha: CVaR threshold

    Returns:
        cvar_values: [n_samples] CVaR 값
    """
    # samples: [n_samples, n_w, 1] → [n_samples, n_w]
    samples = samples.squeeze(-1)

    n_w = samples.shape[1]
    n_worst = max(1, int(n_w * alpha))

    # 각 샘플에 대해 worst n_worst개의 평균 계산
    worst_samples, _ = torch.topk(samples, n_worst, dim=1, largest=False)
    cvar_values = worst_samples.mean(dim=1)

    return cvar_values


def objective_function(X, images_data, yolo_detector, alpha=0.3, verbose=False):
    """
    CVaR 목적 함수

    Args:
        X: [1, 8] 파라미터 (AirLine 6D + RANSAC 2D: Q, QG)
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
    }

    # RANSAC 가중치 (Q, QG 개별)
    ransac_weights = (X[0, 6].item(), X[0, 7].item())

    scores = []

    for i, img_data in enumerate(images_data):
        try:
            image = img_data['image']
            gt_coords = img_data['gt_coords']

            detected_coords = detect_with_full_pipeline(image, params, yolo_detector, ransac_weights)

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
                           n_iterations=30, n_initial=15, alpha=0.3, n_w=15):
    """
    BoRisk 알고리즘 완전 구현

    핵심 원리:
    1. 환경 벡터 사전 추출 (모든 이미지)
    2. w_set 샘플링 (매 iteration마다 n_w개)
    3. GP 모델: (x, w) → y 학습
    4. qMFKG 획득 함수 + CVaR objective
    5. 매 iteration마다 w_set만 평가 (113개 아님!)
    """
    print("\n" + "="*60)
    print(f"BoRisk CVaR Optimization")
    print(f"Total images: {len(images_data)}")
    print(f"CVaR α: {alpha} (worst {int(alpha*100)}%)")
    print(f"w_set size: {n_w}")
    print("="*60)

    # ===== Phase 0: 환경 벡터 사전 추출 =====
    print(f"\n[Phase 0] Environment feature extraction")
    all_env_features = extract_all_environments(images_data)

    # ===== Phase 1: 초기 샘플링 =====
    print(f"\n[Phase 1] Initial sampling ({n_initial} samples)")
    print(f"  - Each sample evaluated on {n_w} environments")
    print(f"  - Total evaluations: {n_initial * n_w}")

    sobol = SobolEngine(dimension=8, scramble=True)  # 8D: AirLine 6D + RANSAC 2D
    X_init = sobol.draw(n_initial).to(dtype=DTYPE, device=DEVICE)
    X_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * X_init

    # 초기 샘플링: 각 x에 대해 w_set 평가
    train_X_params = []  # x만 저장 (9D)
    train_X_full = []    # (x, w) concat (15D)
    train_Y = []
    init_w_indices_list = []  # 각 초기 샘플의 w_indices 저장

    for i, x in enumerate(X_init):
        # w_set 샘플링 (초기에는 고정된 seed 사용)
        w_set, w_indices = sample_w_set(all_env_features, n_w=n_w, seed=i)
        init_w_indices_list.append(w_indices)

        # w_set에 대해서만 평가
        scores = evaluate_on_w_set(x.unsqueeze(0), images_data, yolo_detector, w_indices)

        # 각 (x, w) 쌍을 만들어서 저장
        for j in range(n_w):
            x_w = torch.cat([x, w_set[j]])  # [9+6=15]
            train_X_full.append(x_w)
            train_X_params.append(x.clone())
        train_Y.append(scores)

        cvar = compute_cvar_from_scores(scores.squeeze(-1), alpha)
        print(f"Init {i+1}/{n_initial}: CVaR={cvar:.4f}, mean={scores.mean():.4f}")

    train_X_full = torch.stack(train_X_full)      # [n_initial * n_w, 15]
    train_X_params = torch.stack(train_X_params)  # [n_initial * n_w, 9]
    train_Y = torch.cat(train_Y)                  # [n_initial * n_w, 1]

    # ===== Phase 2: Y 값 정규화 =====
    Y_mean = train_Y.mean()
    Y_std = train_Y.std()

    if Y_std < 1e-6:
        print("\n⚠️ Warning: Low variance in Y values, adding noise...")
        train_Y = train_Y + 0.01 * torch.randn_like(train_Y)
        Y_mean = train_Y.mean()
        Y_std = train_Y.std()

    train_Y_normalized = (train_Y - Y_mean) / Y_std

    print(f"\n[Phase 2] Data normalization")
    print(f"  mean={Y_mean:.4f}, std={Y_std:.4f}")
    print(f"  min={train_Y.min().item():.4f}, max={train_Y.max().item():.4f}")

    # ===== Phase 3: GP 모델 생성 (BoRisk 구조!) =====
    print(f"\n[Phase 3] GP model initialization")
    print(f"  - Input: (x, w) concat [15D = 9D params + 6D env]")
    print(f"  - GP learns: (x, w) → y")
    print(f"  - train_X_full shape: {train_X_full.shape}")
    print(f"  - train_Y shape: {train_Y_normalized.shape}")

    gp = SingleTaskGP(
        train_X_full,         # [N*n_w, 15] (x,w) concat
        train_Y_normalized    # [N*n_w, 1] 정규화된 Y
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    print(f"  GP noise level: {gp.likelihood.noise.item():.6f}")

    # Best CVaR 추적 (각 x에 대해)
    best_cvar_history = []
    for i in range(n_initial):
        start_idx = i * n_w
        end_idx = start_idx + n_w
        cvar = compute_cvar_from_scores(train_Y[start_idx:end_idx].squeeze(), alpha)
        best_cvar_history.append(cvar)
    print(f"  Initial best CVaR: {max(best_cvar_history):.4f}")

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ===== Phase 5: BO 루프 (BoRisk!) =====
    print(f"\n[Phase 4] BO iterations (BoRisk)")
    print("-"*60)

    for iteration in range(n_iterations):
        # 5.1: 새로운 w_set 샘플링 (매 iteration마다)
        w_set, w_indices = sample_w_set(all_env_features, n_w=n_w)

        # 5.2: BoRisk Knowledge Gradient 획득 함수 (논문 구현!)
        try:
            from borisk_kg import optimize_borisk

            # BoRisk Knowledge Gradient 사용
            candidate, acq_value, acq_name = optimize_borisk(
                gp, w_set, BOUNDS, alpha=alpha,
                use_full_kg=True  # 정확한 BoRisk-KG 사용 (판타지 관측)
            )
            print(f"  Using {acq_name}: acq_value={acq_value:.4f}")
            acq_value = torch.tensor(acq_value) if not torch.is_tensor(acq_value) else acq_value

        except ImportError as e:
            print(f"\n⚠️ BoRisk KG import failed: {e}, falling back to UCB...")

            # 폴백: UCB with manual w addition
            try:
                candidates = torch.rand(100, 9, dtype=DTYPE, device=DEVICE)
                candidates = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * candidates

                best_idx = 0
                best_score = -float('inf')
                for i in range(100):
                    x = candidates[i]
                    x_w_list = []
                    for w in w_set:
                        x_w_list.append(torch.cat([x, w]).unsqueeze(0))
                    X_full = torch.cat(x_w_list, dim=0)

                    with torch.no_grad():
                        posterior = gp.posterior(X_full)
                        score = posterior.mean.mean().item()

                    if score > best_score:
                        best_score = score
                        best_idx = i

                candidate = candidates[best_idx].unsqueeze(0)
                acq_value = torch.tensor(best_score)
                acq_name = "UCB-fallback"
            except Exception as e2:
                print(f"\n⚠️ UCB also failed: {e2}")
                candidate = torch.rand(1, 9, dtype=DTYPE, device=DEVICE)
                candidate = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * candidate
                acq_value = torch.tensor(0.0)
                acq_name = "Random"

        # 5.3: w_set에서만 평가 (BoRisk의 핵심!)
        new_scores = evaluate_on_w_set(candidate, images_data, yolo_detector, w_indices)
        new_cvar = compute_cvar_from_scores(new_scores.squeeze(), alpha)

        # 5.4: 정규화
        new_scores_normalized = (new_scores - Y_mean) / (Y_std + 1e-6)

        # 5.5: 로그 저장
        iter_log = {
            "iteration": iteration + 1,
            "acq_function": acq_name,
            "acq_value": float(acq_value.item()) if torch.is_tensor(acq_value) else 0.0,
            "parameters": {
                "edgeThresh1": float(candidate[0, 0].item()),
                "simThresh1": float(candidate[0, 1].item()),
                "pixelRatio1": float(candidate[0, 2].item()),
                "edgeThresh2": float(candidate[0, 3].item()),
                "simThresh2": float(candidate[0, 4].item()),
                "pixelRatio2": float(candidate[0, 5].item()),
                "ransac_weight_q": float(candidate[0, 6].item()),
                "ransac_weight_qg": float(candidate[0, 7].item()),
            },
            "cvar": float(new_cvar),
            "mean_score": float(new_scores.mean().item()),
            "n_w": n_w
        }

        log_file = log_dir / f"iter_{iteration+1:03d}.json"
        with open(log_file, 'w') as f:
            json.dump(iter_log, f, indent=2)

        # 5.6: 데이터 추가 (각 (x, w) 쌍 추가)
        for j in range(n_w):
            x_w = torch.cat([candidate[0], w_set[j]]).unsqueeze(0)
            train_X_full = torch.cat([train_X_full, x_w])
            train_X_params = torch.cat([train_X_params, candidate])
        train_Y = torch.cat([train_Y, new_scores])
        train_Y_normalized = torch.cat([train_Y_normalized, new_scores_normalized])

        best_cvar_history.append(new_cvar)

        # 5.7: GP 재학습
        try:
            gp = SingleTaskGP(
                train_X_full,
                train_Y_normalized
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"  ⚠️ GP refit failed: {e}")
            # 노이즈 추가 후 재시도
            train_Y_normalized = train_Y_normalized + 0.01 * torch.randn_like(train_Y_normalized)
            gp = SingleTaskGP(
                train_X_full,
                train_Y_normalized
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        # 5.8: 진행 상황 출력
        current_best = max(best_cvar_history)
        improvement = new_cvar - current_best if len(best_cvar_history) > 1 else 0.0

        print(f"Iter {iteration+1}/{n_iterations} ({acq_name}): CVaR={new_cvar:.4f}, Best={current_best:.4f} ({improvement:+.4f})")

        # 5.9: 조기 종료 체크
        if iteration > 10:
            recent_improvements = [best_cvar_history[i] - best_cvar_history[i-1]
                                  for i in range(-5, 0)]
            if all(abs(imp) < 1e-5 for imp in recent_improvements):
                print("\n조기 종료: 최근 5회 개선 없음")
                break

    # ===== Phase 6: 최종 결과 =====
    # Best CVaR를 가진 x 찾기
    best_idx = best_cvar_history.index(max(best_cvar_history))
    if best_idx < n_initial:
        # 초기 샘플
        best_X = X_init[best_idx]
    else:
        # BO iteration
        bo_idx = best_idx - n_initial
        best_X = train_X_params[n_initial * n_w + bo_idx * n_w]  # 첫 번째 w에서의 x

    print("\n" + "="*60)
    print("BoRisk Optimization Complete")
    print(f"Best CVaR: {max(best_cvar_history):.4f}")
    print(f"Found at iteration: {best_idx + 1}")
    print("="*60)

    return best_X, best_cvar_history, train_Y


if __name__ == "__main__":
    import sys
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description="BoRisk CVaR Optimization for Welding")
    parser.add_argument("--image_dir", default="../dataset/images/test", help="Image directory")
    parser.add_argument("--gt_file", default="../dataset/ground_truth.json", help="Ground truth JSON")
    parser.add_argument("--yolo_model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--iterations", type=int, default=20, help="BO iterations")
    parser.add_argument("--n_initial", type=int, default=10, help="Initial samples")
    parser.add_argument("--n_w", type=int, default=15, help="w_set size (BoRisk)")
    parser.add_argument("--alpha", type=float, default=0.3, help="CVaR alpha (worst percent)")
    parser.add_argument("--complete_only", action="store_true", help="Use only complete GT images")
    parser.add_argument("--n_augment", type=int, default=0, help="Number of augmentations per image")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images (for fast testing)")
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

    # 이미지 개수 제한 (빠른 테스트용)
    if args.max_images is not None and args.max_images < len(images_data):
        import random
        random.seed(42)
        images_data = random.sample(images_data, args.max_images)
        print(f"[INFO] Limited to {args.max_images} images for fast testing")

    # 원본과 증강 이미지 개수 출력
    n_original = sum(1 for img in images_data if not img.get('is_augmented', False))
    n_augmented = len(images_data) - n_original

    print(f"Loaded {n_original} original images")
    if n_augmented > 0:
        print(f"Generated {n_augmented} augmented images (total: {len(images_data)})")
    
    # YOLO 검출기 초기화
    print(f"\nInitializing YOLO detector...")
    yolo_detector = YOLODetector(args.yolo_model)
    
    # BoRisk 최적화 실행
    best_params, history, all_Y = optimize_risk_aware_bo(
        images_data,
        yolo_detector,
        n_iterations=args.iterations,
        n_initial=args.n_initial,
        alpha=args.alpha,
        n_w=args.n_w
    )
    
    # 결과 출력
    print("\n최적 파라미터:")
    print(f"  edgeThresh1:      {best_params[0]:7.2f}")
    print(f"  simThresh1:       {best_params[1]:7.4f}")
    print(f"  pixelRatio1:      {best_params[2]:7.4f}")
    print(f"  edgeThresh2:      {best_params[3]:7.2f}")
    print(f"  simThresh2:       {best_params[4]:7.4f}")
    print(f"  pixelRatio2:      {best_params[5]:7.4f}")
    print(f"  ransac_weight_q:  {best_params[6]:7.2f}")
    print(f"  ransac_weight_qg: {best_params[7]:7.2f}")
    
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
        "algorithm": "BoRisk",
        "alpha": args.alpha,
        "n_w": args.n_w,
        "n_images": len(images_data),
        "n_original": n_original,
        "n_augmented": n_augmented,
        "iterations": args.iterations,
        "n_initial": args.n_initial,
        "best_params": {
            "edgeThresh1": float(best_params[0]),
            "simThresh1": float(best_params[1]),
            "pixelRatio1": float(best_params[2]),
            "edgeThresh2": float(best_params[3]),
            "simThresh2": float(best_params[4]),
            "pixelRatio2": float(best_params[5]),
            "ransac_weight_q": float(best_params[6]),
            "ransac_weight_qg": float(best_params[7]),
        },
        "history": [float(x) for x in history],
        "final_cvar": float(history[-1]),
        "initial_cvar": float(history[0]),
        "improvement": float((history[-1] - history[0]) / (history[0] + 1e-6) * 100),
        "complete_only": args.complete_only,
        "n_augment": args.n_augment
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n결과 저장: {result_file}")
    print("="*60)