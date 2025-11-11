"""
BoRisk Bayesian Optimization with CVaR - FIXED VERSION

수정사항:
1. GT 없는 이미지 필터링
2. 에러 처리 개선
3. 더 자세한 디버깅 정보
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
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from tqdm import tqdm
from scipy.spatial.distance import cdist

# 프로젝트 모듈
sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo_detector import YOLODetector
from full_pipeline import detect_with_full_pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

BOUNDS_10D = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 0.0, 0.0],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 1.0, 1.0]
], dtype=DTYPE, device=DEVICE)

BOUNDS_6D = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15]
], dtype=DTYPE, device=DEVICE)


def simple_line_evaluation(detected_coords, gt_coords, image_size=(640, 480), 
                          angle_weight=0.9, distance_weight=0.1):
    """
    선분 기반 평가: 각도 유사도 + 거리 유사도 (초엄격!)
    
    Args:
        detected_coords: 검출된 좌표 dict
        gt_coords: GT 좌표 dict
        image_size: (width, height) 정규화용
        angle_weight: 각도 가중치 (0.9 = 90%) ← 각도 최우선!
        distance_weight: 거리 가중치 (0.1 = 10%)
    
    Returns:
        score: float [0, 1]
    """
    # 3개 선분: longi_left, longi_right, collar_left
    line_definitions = [
        ('longi_left', 'longi_left_lower_x', 'longi_left_lower_y',
                      'longi_left_upper_x', 'longi_left_upper_y'),
        ('longi_right', 'longi_right_lower_x', 'longi_right_lower_y',
                       'longi_right_upper_x', 'longi_right_upper_y'),
        ('collar_left', 'collar_left_lower_x', 'collar_left_lower_y',
                       'collar_left_upper_x', 'collar_left_upper_y')
    ]
    
    # 이미지 대각선 (거리 정규화용)
    diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
    distance_threshold = diagonal * 0.01  # ← 1%! (초엄격!)
    
    line_scores = []
    
    for name, x1_key, y1_key, x2_key, y2_key in line_definitions:
        # GT 선분
        gt_x1 = gt_coords.get(x1_key, 0)
        gt_y1 = gt_coords.get(y1_key, 0)
        gt_x2 = gt_coords.get(x2_key, 0)
        gt_y2 = gt_coords.get(y2_key, 0)
        
        # 검출 선분
        det_x1 = detected_coords.get(x1_key, 0)
        det_y1 = detected_coords.get(y1_key, 0)
        det_x2 = detected_coords.get(x2_key, 0)
        det_y2 = detected_coords.get(y2_key, 0)
        
        # GT가 없으면 스킵
        if gt_x1 == 0 and gt_y1 == 0 and gt_x2 == 0 and gt_y2 == 0:
            continue
        
        # 검출이 없으면 0점
        if det_x1 == 0 and det_y1 == 0 and det_x2 == 0 and det_y2 == 0:
            line_scores.append(0.0)
            continue
        
        # === 1. 각도 유사도 ===
        # GT 선분 각도
        gt_angle = np.arctan2(gt_y2 - gt_y1, gt_x2 - gt_x1)
        # 검출 선분 각도
        det_angle = np.arctan2(det_y2 - det_y1, det_x2 - det_x1)
        
        # 각도 차이 (0 ~ π)
        angle_diff = abs(gt_angle - det_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # 각도 유사도 (0도 = 1.0, 180도 = 0.0)
        angle_similarity = 1.0 - (angle_diff / np.pi)
        
        # === 2. 거리 유사도 ===
        # 양 끝점의 평균 거리
        dist_p1 = np.sqrt((det_x1 - gt_x1)**2 + (det_y1 - gt_y1)**2)
        dist_p2 = np.sqrt((det_x2 - gt_x2)**2 + (det_y2 - gt_y2)**2)
        avg_distance = (dist_p1 + dist_p2) / 2
        
        # 거리 유사도 (가까울수록 높음)
        distance_similarity = max(0.0, 1.0 - (avg_distance / distance_threshold))
        
        # === 3. 종합 점수 ===
        line_score = (angle_weight * angle_similarity + 
                     distance_weight * distance_similarity)
        
        line_scores.append(line_score)
    
    if len(line_scores) == 0:
        return 0.0
    
    # 평균 점수
    score = np.mean(line_scores)
    
    return float(np.clip(score, 0.0, 1.0))


def find_closest_environment_image(w_query, images_data, env_keys):
    """환경 벡터 w에 가장 가까운 실제 이미지 찾기"""
    if isinstance(w_query, torch.Tensor):
        w_query = w_query.cpu().numpy()
    
    env_vectors = []
    for img in images_data:
        env = img['environment']
        vec = [env.get(k, 0.5) for k in env_keys]
        env_vectors.append(vec)
    
    env_vectors = np.array(env_vectors)
    distances = cdist([w_query], env_vectors, metric='euclidean')[0]
    closest_idx = distances.argmin()
    
    return images_data[closest_idx], distances[closest_idx]


def evaluate_at_xw(x_params, img_data, yolo_detector, verbose=False):
    """
    특정 (x, w) 지점에서 평가 (실제 관측)
    
    Args:
        x_params: [6D] 파라미터 텐서
        img_data: 이미지 데이터
        yolo_detector: YOLO 검출기
        verbose: 디버그 출력
    
    Returns:
        score: float
    """
    params = {
        'edgeThresh1': x_params[0].item(),
        'simThresh1': x_params[1].item(),
        'pixelRatio1': x_params[2].item(),
        'edgeThresh2': x_params[3].item(),
        'simThresh2': x_params[4].item(),
        'pixelRatio2': x_params[5].item(),
    }
    
    image = img_data['image']
    image_name = img_data['name']
    gt_coords = img_data['gt_coords']
    
    try:
        # 파이프라인 실행
        detected_coords = detect_with_full_pipeline(image, params, yolo_detector)
        
        # 거리 기반 평가 (이미지 크기 전달)
        h, w = image.shape[:2]
        score = simple_line_evaluation(detected_coords, gt_coords, image_size=(w, h))
        
        if verbose:
            det_count = sum(1 for v in detected_coords.values() if v != 0)
            gt_count = sum(1 for v in gt_coords.values() if v != 0)
            print(f"      검출: {det_count}/12, GT: {gt_count}/12, 점수: {score:.3f}")
        
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if verbose:
            print(f"[ERROR] Evaluation failed for {image_name}: {e}")
            traceback.print_exc()
        score = 0.0
    
    return score


def compute_cvar_from_gp(gp_model, x_params, env_samples, alpha=0.1):
    """GP 모델로부터 CVaR 계산"""
    x_expanded = x_params.repeat(env_samples.shape[0], 1)
    xw_samples = torch.cat([x_expanded, env_samples], dim=1)
    
    with torch.no_grad():
        posterior = gp_model.posterior(xw_samples)
        mean = posterior.mean.squeeze(-1)
    
    n_worst = max(1, int(env_samples.shape[0] * alpha))
    worst_scores, _ = torch.topk(mean, n_worst, largest=False)
    cvar = worst_scores.mean()
    
    return cvar.item()


class BoRiskCVaRKG:
    """Custom Knowledge Gradient for CVaR maximization"""
    def __init__(self, gp_model, alpha=0.1, num_fantasies=32, 
                 env_samples_per_eval=50, env_keys=None):
        self.gp = gp_model
        self.alpha = alpha
        self.num_fantasies = num_fantasies
        self.env_samples_per_eval = env_samples_per_eval
        self.env_keys = env_keys or ['brightness', 'contrast', 'edge_density', 'texture_complexity']
    
    def __call__(self, X):
        """획득함수 계산"""
        X = X.to(dtype=DTYPE, device=DEVICE)
        batch_size = X.shape[0]
        acq_values = []
        
        env_samples = torch.rand(self.env_samples_per_eval, 4, 
                                dtype=DTYPE, device=DEVICE)
        
        for i in range(batch_size):
            x_cand = X[i, :6]
            xw_cand = X[i:i+1]
            
            with torch.no_grad():
                posterior = self.gp.posterior(xw_cand)
                mu = posterior.mean
                sigma = posterior.stddev
            
            fantasy_ys = mu + sigma * torch.randn(self.num_fantasies, 1, 
                                                  dtype=DTYPE, device=DEVICE)
            
            improvements = []
            
            for y_fantasy in fantasy_ys:
                try:
                    fantasy_gp = self.gp.condition_on_observations(
                        X=xw_cand,
                        Y=y_fantasy.reshape(1, 1)
                    )
                    
                    cvar_new = compute_cvar_from_gp(fantasy_gp, x_cand, env_samples, self.alpha)
                    cvar_old = compute_cvar_from_gp(self.gp, x_cand, env_samples, self.alpha)
                    
                    improvement = cvar_new - cvar_old
                    improvements.append(improvement)
                    
                except:
                    improvements.append(0.0)
            
            expected_improvement = np.mean(improvements)
            acq_values.append(expected_improvement)
        
        return torch.tensor(acq_values, dtype=DTYPE, device=DEVICE)


def load_dataset_with_environment(image_dir, gt_file, env_file, filter_incomplete=True):
    """
    데이터셋 + 환경변수 로드
    
    Args:
        filter_incomplete: True이면 GT가 불완전한 이미지 제거
    """
    image_dir = Path(image_dir)
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    with open(env_file, 'r') as f:
        env_data = json.load(f)
    
    images_data = []
    skipped_count = 0
    
    for img_name, gt in gt_data.items():
        # 이미지 로드
        img_path = None
        for ext in ['.jpg', '.png']:
            p = image_dir / f"{img_name}{ext}"
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            skipped_count += 1
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            skipped_count += 1
            continue
        
        # GT 좌표 확인
        gt_coords = gt.get('coordinates', gt)
        
        if filter_incomplete:
            # 최소한의 좌표가 있는지 확인
            required_keys = [
                'longi_left_lower_x', 'longi_left_lower_y',
                'longi_right_lower_x', 'longi_right_lower_y'
            ]
            if not all(gt_coords.get(k, 0) != 0 for k in required_keys):
                skipped_count += 1
                continue
        
        # 환경변수 매칭
        env = env_data.get(img_name, {})
        
        img_data = {
            'name': img_name,
            'image': image,
            'gt_coords': gt_coords,
            'environment': env
        }
        images_data.append(img_data)
    
    print(f"  로드됨: {len(images_data)}장")
    if skipped_count > 0:
        print(f"  스킵됨: {skipped_count}장 (GT 불완전 또는 이미지 없음)")
    
    return images_data


def compute_true_cvar_on_dataset(x_params, images_data, yolo_detector, alpha=0.1, verbose=False):
    """전체 데이터셋에 대한 진짜 CVaR 계산 (검증용)"""
    scores = []
    
    iterator = tqdm(images_data, desc="  Computing true CVaR", leave=False) if verbose else images_data
    
    for img_data in iterator:
        score = evaluate_at_xw(x_params, img_data, yolo_detector, verbose=False)
        scores.append(score)
    
    scores = np.array(scores)
    n_worst = max(1, int(len(scores) * alpha))
    worst_scores = np.sort(scores)[:n_worst]
    cvar = np.mean(worst_scores)
    
    if verbose:
        print(f"    전체 점수: mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f}")
    
    return cvar


def optimize_borisk_cvar(images_data, yolo_detector, alpha=0.1, 
                         n_iterations=30, n_initial=10,
                         num_fantasies=32, validate_every=0,
                         checkpoint_dir="checkpoints"):
    """
    BoRisk CVaR 최적화 메인 함수
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    env_keys = ['brightness', 'contrast', 'edge_density', 'texture_complexity']
    
    print("\n" + "="*70)
    print(f"BoRisk CVaR Optimization (FIXED)")
    print(f"Images: {len(images_data)}")
    print(f"CVaR α: {alpha} (worst {int(alpha*100)}%)")
    print(f"Fantasies: {num_fantasies}")
    print("="*70)
    
    # 초기 샘플링
    print(f"\n[초기 샘플링: {n_initial}개]")
    sobol = SobolEngine(dimension=10, scramble=True)
    XW_init_unit = sobol.draw(n_initial).to(dtype=DTYPE, device=DEVICE)
    XW_init = BOUNDS_10D[0] + (BOUNDS_10D[1] - BOUNDS_10D[0]) * XW_init_unit
    
    X_train = []
    Y_train = []
    
    for i, xw in enumerate(XW_init):
        x_params = xw[:6]
        w_env = xw[6:]
        
        print(f"\nInitial {i+1}/{n_initial}")
        print(f"  x: edge=[{x_params[0]:.1f}, {x_params[3]:.1f}], "
              f"sim=[{x_params[1]:.3f}, {x_params[4]:.3f}]")
        print(f"  w: [{w_env[0]:.3f}, {w_env[1]:.3f}, {w_env[2]:.3f}, {w_env[3]:.3f}]")
        
        # 가장 가까운 이미지 찾기
        img_data, dist = find_closest_environment_image(w_env, images_data, env_keys)
        print(f"  → 이미지: {img_data['name']} (dist={dist:.3f})")
        
        # 평가
        score = evaluate_at_xw(x_params, img_data, yolo_detector, verbose=True)
        print(f"  → 점수: {score:.4f}")
        
        X_train.append(xw)
        Y_train.append(score)
    
    X_train = torch.stack(X_train)
    Y_train = torch.tensor(Y_train, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    
    # GP 학습
    print("\n[GP 초기 학습]")
    gp = SingleTaskGP(X_train, Y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    best_idx = Y_train.argmax()
    best_X = X_train[best_idx, :6]
    
    print(f"  초기 최고 점수: {Y_train[best_idx].item():.4f}")
    
    # BO 루프
    history = {
        'observed_scores': Y_train.squeeze(-1).cpu().numpy().tolist(),
        'best_params_history': [best_X.cpu().numpy().tolist()],
        'true_cvar_history': []
    }
    
    print(f"\n{'='*70}")
    print("BO 반복 시작")
    print(f"{'='*70}\n")
    
    for iteration in range(n_iterations):
        print(f"\n[Iteration {iteration+1}/{n_iterations}]")
        
        # 1. 획득함수
        print("  [1/4] 획득함수 생성...")
        kg_acq = BoRiskCVaRKG(
            gp, 
            alpha=alpha, 
            num_fantasies=num_fantasies,
            env_keys=env_keys
        )
        
        # 2. 최적화
        print("  [2/4] 획득함수 최적화...")
        try:
            candidate, acq_value = optimize_acqf(
                kg_acq,
                bounds=BOUNDS_10D,
                q=1,
                num_restarts=5,
                raw_samples=256
            )
        except Exception as e:
            print(f"  [WARN] 획득함수 최적화 실패, 랜덤 샘플링: {e}")
            candidate = torch.rand(1, 10, dtype=DTYPE, device=DEVICE)
            candidate = BOUNDS_10D[0] + (BOUNDS_10D[1] - BOUNDS_10D[0]) * candidate
        
        x_next = candidate[0, :6]
        w_next = candidate[0, 6:]
        
        print(f"    제안: edge=[{x_next[0]:.1f}, {x_next[3]:.1f}], "
              f"sim=[{x_next[1]:.3f}, {x_next[4]:.3f}]")
        
        # 3. 실제 관측
        print("  [3/4] 실제 평가...")
        img_next, dist = find_closest_environment_image(w_next, images_data, env_keys)
        print(f"    이미지: {img_next['name']} (dist={dist:.3f})")
        
        y_next = evaluate_at_xw(x_next, img_next, yolo_detector, verbose=True)
        print(f"    → 점수: {y_next:.4f}")
        
        # 4. GP 업데이트
        print("  [4/4] GP 업데이트...")
        X_train = torch.cat([X_train, candidate])
        Y_train = torch.cat([Y_train, torch.tensor([[y_next]], dtype=DTYPE, device=DEVICE)])
        
        gp = SingleTaskGP(X_train, Y_train)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        best_idx = Y_train.argmax()
        best_X = X_train[best_idx, :6]
        
        print(f"  현재 최고 관측: {Y_train[best_idx].item():.4f}")
        
        history['observed_scores'].append(y_next)
        history['best_params_history'].append(best_X.cpu().numpy().tolist())
        
        # 검증 (옵션)
        if validate_every > 0 and (iteration + 1) % validate_every == 0:
            print(f"\n  [검증] 전체 데이터셋에서 CVaR 계산...")
            true_cvar = compute_true_cvar_on_dataset(best_X, images_data, yolo_detector, alpha, verbose=True)
            print(f"  ✓ True CVaR: {true_cvar:.4f}")
            history['true_cvar_history'].append({
                'iteration': iteration + 1,
                'cvar': true_cvar
            })
        
        # 체크포인트
        checkpoint = {
            'iteration': iteration + 1,
            'X_train': X_train.cpu().numpy(),
            'Y_train': Y_train.cpu().numpy(),
            'best_X': best_X.cpu().numpy(),
            'history': history
        }
        torch.save(checkpoint, checkpoint_path / f"iter_{iteration+1}.pt")
    
    print("\n" + "="*70)
    print("최적화 완료")
    print("="*70)
    
    return best_X, history


if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description="BoRisk CVaR Optimization (FIXED)")
    parser.add_argument("--image_dir", default="dataset/images/test")
    parser.add_argument("--gt_file", default="dataset/ground_truth.json")
    parser.add_argument("--env_file", default="environment_independent.json")
    parser.add_argument("--yolo_model", default="models/best.pt")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--n_initial", type=int, default=10)
    parser.add_argument("--num_fantasies", type=int, default=32)
    parser.add_argument("--validate_every", type=int, default=0, help="검증 주기 (0=안함)")
    parser.add_argument("--checkpoint_dir", default="checkpoints_borisk")
    parser.add_argument("--filter_incomplete", action="store_true", help="GT 불완전한 이미지 제거")
    args = parser.parse_args()
    
    # 데이터 로드
    print(f"Loading dataset...")
    images_data = load_dataset_with_environment(
        args.image_dir, 
        args.gt_file, 
        args.env_file,
        filter_incomplete=args.filter_incomplete
    )
    
    if len(images_data) == 0:
        print("[ERROR] 사용 가능한 이미지가 없습니다!")
        sys.exit(1)
    
    # YOLO 초기화
    print(f"Initializing YOLO...")
    yolo_detector = YOLODetector(args.yolo_model)
    
    # 최적화
    best_params, history = optimize_borisk_cvar(
        images_data,
        yolo_detector,
        alpha=args.alpha,
        n_iterations=args.iterations,
        n_initial=args.n_initial,
        num_fantasies=args.num_fantasies,
        validate_every=args.validate_every,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 결과 출력
    print("\n최적 파라미터:")
    print(f"  edgeThresh1:  {best_params[0]:7.2f}")
    print(f"  simThresh1:   {best_params[1]:7.4f}")
    print(f"  pixelRatio1:  {best_params[2]:7.4f}")
    print(f"  edgeThresh2:  {best_params[3]:7.2f}")
    print(f"  simThresh2:   {best_params[4]:7.4f}")
    print(f"  pixelRatio2:  {best_params[5]:7.4f}")
    
    # 결과 저장
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"borisk_cvar_fixed_{timestamp}.json"
    
    result_data = {
        "alpha": args.alpha,
        "n_images": len(images_data),
        "iterations": args.iterations,
        "num_fantasies": args.num_fantasies,
        "best_params": {
            "edgeThresh1": float(best_params[0]),
            "simThresh1": float(best_params[1]),
            "pixelRatio1": float(best_params[2]),
            "edgeThresh2": float(best_params[3]),
            "simThresh2": float(best_params[4]),
            "pixelRatio2": float(best_params[5]),
        },
        "history": history
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n결과 저장: {result_file}")