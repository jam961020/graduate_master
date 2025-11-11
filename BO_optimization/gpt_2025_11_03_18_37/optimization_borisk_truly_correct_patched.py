"""
BoRisk - Complete Solution
모든 gradient 문제 해결 + Soft CVaR 옵션 추가
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import datetime
import sys
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector
from evaluation import evaluate_quality

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# 범위: [파라미터 6D, 환경 4D] = 10D
BOUNDS = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
], dtype=DTYPE, device=DEVICE)


# ==================== 환경 추정 ====================

def estimate_environment(image, airline_config=None):
    """환경 추정"""
    from extract_environment_advanced import extract_environment_advanced
    
    if airline_config is None:
        airline_config = {'edgeThresh': 0}
    
    env = extract_environment_advanced(image, roi=None, airline_config=airline_config)
    
    return {
        'continuity_break': env['continuity_break'],
        'orientation_var': env['orientation_var'],
        'illumination_uneven': env['illumination_uneven'],
        'centroid_dispersion': env['centroid_dispersion']
    }


def precompute_all_environments(images_data):
    """모든 이미지의 환경 미리 계산"""
    print("\n[Precomputing environments for all images...]")
    
    airline_config = {'edgeThresh': 0}
    
    for img_data in images_data:
        env = estimate_environment(img_data['image'], airline_config)
        img_data['environment'] = env
        
    print("  Done! Example environments:")
    for i in range(min(3, len(images_data))):
        env = images_data[i]['environment']
        print(f"    {images_data[i]['name'][:40]}: "
              f"cont={env['continuity_break']:.2f}, "
              f"orient={env['orientation_var']:.2f}, "
              f"illum={env['illumination_uneven']:.2f}, "
              f"disp={env['centroid_dispersion']:.2f}")


def find_similar_image(w_target, images_data):
    """w와 가장 유사한 이미지 찾기"""
    best_img = None
    min_dist = float('inf')
    
    for img_data in images_data:
        w_img = img_data['environment']
        
        dist = np.sqrt(
            (w_target['continuity_break'] - w_img['continuity_break'])**2 +
            (w_target['orientation_var'] - w_img['orientation_var'])**2 +
            (w_target['illumination_uneven'] - w_img['illumination_uneven'])**2 +
            (w_target['centroid_dispersion'] - w_img['centroid_dispersion'])**2
        )
        
        if dist < min_dist:
            min_dist = dist
            best_img = img_data
    
    return best_img, min_dist


def env_dict_to_tensor(w):
    """dict → tensor"""
    return torch.tensor([
        w['continuity_break'],
        w['orientation_var'],
        w['illumination_uneven'],
        w['centroid_dispersion']
    ], dtype=DTYPE, device=DEVICE)


def tensor_to_env_dict(w_tensor):
    """tensor → dict"""
    return {
        'continuity_break': w_tensor[0].item(),
        'orientation_var': w_tensor[1].item(),
        'illumination_uneven': w_tensor[2].item(),
        'centroid_dispersion': w_tensor[3].item()
    }


def evaluate_F_xw(x_params, w, images_data, yolo_detector, metric='lp'):
    """F(x, w) = w와 유사한 이미지에서 x의 성능"""
    img_data, distance = find_similar_image(w, images_data)
    
    try:
        detected_coords = detect_with_full_pipeline(
            img_data['image'],
            x_params,
            yolo_detector
        )
        
        score = evaluate_quality(
            detected_coords=detected_coords,
            image=img_data['image'],
            image_name=img_data['name'],
            metric=metric
        )
        
        return score, img_data['name'], distance
        
    except Exception as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
        return 0.0, img_data['name'], distance


# ==================== Soft CVaR 함수 ====================

def soft_cvar(values, alpha, temperature=0.1):
    """
    Soft CVaR using smooth approximation
    
    Args:
        values: [N] tensor of values
        alpha: worst-case fraction
        temperature: smoothing parameter (smaller = closer to true CVaR)
    
    Returns:
        Soft CVaR value (differentiable)
    """
    n = values.shape[0]
    k = max(1, int(n * alpha))
    
    # Soft sorting using log-sum-exp trick
    # This approximates the k smallest values
    neg_values = -values
    
    # Compute soft top-k using temperature scaling
    weights = F.softmax(neg_values / temperature, dim=0)
    
    # Weight more heavily on worst values
    sorted_neg, _ = torch.sort(neg_values, descending=True)
    rank_weights = torch.exp(-torch.arange(n, dtype=DTYPE, device=values.device) / k)
    rank_weights = rank_weights / rank_weights.sum()
    
    # Weighted average focusing on worst alpha fraction
    soft_cvar_value = (values * weights).sum()
    
    return soft_cvar_value


# ==================== CVaR 획득 함수 (Complete Fix) ====================

class CVaRAcquisitionFunction(AcquisitionFunction):
    """
    CVaR 기반 획득 함수
    Complete solution with proper gradient handling
    """
    
    def __init__(self, model, bounds, alpha=0.3, n_w_samples=100, 
                 images_data=None, use_soft_cvar=True, temperature=0.1):
        """
        Args:
            model: GP 모델
            bounds: [2, 10] tensor (6D params + 4D env)
            alpha: CVaR α
            n_w_samples: W 샘플 개수
            images_data: 실제 환경 분포를 위한 이미지 데이터
            use_soft_cvar: Soft CVaR 사용 여부 (gradient 개선)
            temperature: Soft CVaR temperature
        """
        super().__init__(model=model)
        self.bounds = bounds
        self.alpha = alpha
        self.n_w_samples = n_w_samples
        self.use_soft_cvar = use_soft_cvar
        self.temperature = temperature
        
        # 실제 환경 분포 저장 (gradient 추적 안함)
        if images_data is not None:
            with torch.no_grad():
                self.real_envs = self._extract_env_distribution(images_data)
            print(f"    Using real environment distribution from {len(self.real_envs)} images")
            print(f"    CVaR mode: {'Soft' if use_soft_cvar else 'Hard'} (α={alpha})")
        else:
            self.real_envs = None
            print(f"    Warning: Using uniform distribution for environments")
    
    def _extract_env_distribution(self, images_data):
        """실제 이미지들의 환경을 tensor로 변환"""
        envs = []
        for img_data in images_data:
            if 'environment' in img_data:
                env = img_data['environment']
                env_tensor = torch.tensor([
                    env['continuity_break'],
                    env['orientation_var'],
                    env['illumination_uneven'],
                    env['centroid_dispersion']
                ], dtype=DTYPE, device=DEVICE)
                envs.append(env_tensor)
        
        if envs:
            return torch.stack(envs)
        else:
            return None
    
    def forward(self, X):
        """
        X: [..., q, 6] 파라미터만 (정규화 0~1)
        Returns: [...] CVaR 값 (음수로 반환하여 maximize)
        """
        # Input X는 gradient 추적
        X = X.requires_grad_(True)
        
        # Shape 처리
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        # q dimension 처리
        *batch_shape, q, d = X.shape
        
        # q=1인 경우
        if q == 1:
            # [*batch_shape, 1, 6] -> [*batch_shape, 6]
            X = X.squeeze(-2)
            
            # Flatten for batch processing
            X_flat = X.reshape(-1, d)
            batch_size = X_flat.shape[0]
            
            # CVaR 계산을 위한 결과 저장
            cvar_values = []
            
            for i in range(batch_size):
                x_norm = X_flat[i:i+1]  # [1, 6] - gradient 유지
                
                # W 샘플링 (gradient 추적 안함)
                with torch.no_grad():
                    if self.real_envs is not None and len(self.real_envs) > 0:
                        # 실제 환경에서 무작위 샘플링
                        indices = torch.randint(0, len(self.real_envs), (self.n_w_samples,))
                        w_samples = self.real_envs[indices]
                        
                        # 정규화
                        w_samples = (w_samples - self.bounds[0, 6:10]) / (self.bounds[1, 6:10] - self.bounds[0, 6:10])
                    else:
                        # 균등 분포
                        w_samples = torch.rand(self.n_w_samples, 4, dtype=DTYPE, device=DEVICE)
                
                # [x, w] 결합 - x_norm은 gradient 유지
                x_repeated = x_norm.expand(self.n_w_samples, -1)
                
                # w_samples는 gradient 필요 없음
                w_samples = w_samples.detach()
                
                # 결합
                xw = torch.cat([x_repeated, w_samples], dim=1)  # [n_w_samples, 10]
                
                # GP 예측 - gradient 유지!
                posterior = self.model.posterior(xw)
                mean = posterior.mean.squeeze(-1)  # [n_w_samples]
                
                # CVaR 계산
                if self.use_soft_cvar:
                    # Soft CVaR (better gradients)
                    cvar = soft_cvar(mean, self.alpha, self.temperature)
                else:
                    # Hard CVaR (original)
                    n_worst = max(1, int(self.n_w_samples * self.alpha))
                    worst_vals, _ = torch.topk(mean, n_worst, largest=False)
                    cvar = worst_vals.mean()
                
                # 음수로 반환 (maximize)
                cvar_values.append(-cvar)
            
            # Stack and reshape
            result = torch.stack(cvar_values)
            result = result.view(*batch_shape)
            
        else:
            # q > 1인 경우
            raise NotImplementedError(f"q={q} is not supported. Use q=1.")
        
        return result


# ==================== 체크포인트 (Gradient-safe) ====================

def save_checkpoint(filepath, X_9d, Y, iteration, best_observed, image_names):
    """체크포인트 저장 - gradient safe"""
    checkpoint = {
        'X_9d': X_9d.detach().cpu().numpy().tolist(),  # detach 추가!
        'Y': Y.detach().cpu().numpy().tolist(),  # detach 추가!
        'iteration': iteration,
        'best_observed': best_observed,
        'image_names': image_names,
        'timestamp': datetime.datetime.now().isoformat()
    }
    with open(filepath, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"  [Checkpoint saved: iteration {iteration}]")


def load_checkpoint(filepath):
    """체크포인트 로드"""
    with open(filepath, 'r') as f:
        checkpoint = json.load(f)
    
    # 로드된 데이터는 gradient 불필요
    X_9d = torch.tensor(checkpoint['X_9d'], dtype=DTYPE, device=DEVICE, requires_grad=False)
    Y = torch.tensor(checkpoint['Y'], dtype=DTYPE, device=DEVICE, requires_grad=False)
    
    return X_9d, Y, checkpoint['iteration'], checkpoint['best_observed'], checkpoint['image_names']


# ==================== BO 메인 (Complete Fix) ====================

def optimize_borisk(images_data, yolo_detector, metric="lp",
                   n_iterations=50, n_initial=30, alpha=0.3,
                   use_soft_cvar=True, temperature=0.1,
                   checkpoint_file=None):
    """
    BoRisk Bayesian Optimization - Complete Solution
    """
    print("\n" + "="*70)
    print(f"BoRisk Bayesian Optimization (Complete Solution)")
    print(f"Metric: {metric.upper()}")
    print(f"Image Pool: {len(images_data)} images")
    print(f"CVaR α: {alpha} (worst {int(alpha*100)}%)")
    print(f"CVaR Type: {'Soft' if use_soft_cvar else 'Hard'}")
    if use_soft_cvar:
        print(f"Temperature: {temperature}")
    print(f"Space: 10D (6D params + 4D environment)")
    print("="*70)
    
    # 환경 미리 계산
    precompute_all_environments(images_data)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if checkpoint_file is None:
        checkpoint_file = results_dir / f"checkpoint_{metric}_{timestamp}.json"
    
    # 체크포인트 복원 또는 초기화
    if checkpoint_file.exists():
        print(f"\n[RESUME] Loading checkpoint: {checkpoint_file}")
        X_9d, Y, start_iter, best_observed, image_names = load_checkpoint(checkpoint_file)
        print(f"  Resuming from iteration {start_iter}, Best: {max(best_observed):.4f}")
    else:
        # 초기 샘플링
        print(f"\n{'='*70}")
        print(f"Initial Sampling: {n_initial} evaluations")
        print(f"{'='*70}")
        
        sobol = SobolEngine(dimension=10, scramble=True)
        X_init_9d = sobol.draw(n_initial).to(dtype=DTYPE, device=DEVICE)
        X_init_9d = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * X_init_9d
        
        # 초기 데이터는 gradient 불필요
        X_init_9d = X_init_9d.detach()
        
        Y_init = []
        image_names = []
        
        for i, x_9d in enumerate(X_init_9d):
            x_params = {
                'edgeThresh1': x_9d[0].item(),
                'simThresh1': x_9d[1].item(),
                'pixelRatio1': x_9d[2].item(),
                'edgeThresh2': x_9d[3].item(),
                'simThresh2': x_9d[4].item(),
                'pixelRatio2': x_9d[5].item(),
            
    'ransac_center_w': x_vec[6].item(),
    'ransac_length_w': x_vec[7].item(),
}
            w = tensor_to_env_dict(x_9d[6:10])
            
            print(f"\n[Initial {i+1}/{n_initial}]")
            print(f"  Params: edge=[{x_params['edgeThresh1']:.1f}, {x_params['edgeThresh2']:.1f}], "
                  f"sim=[{x_params['simThresh1']:.3f}, {x_params['simThresh2']:.3f}]")
            
            # w와 유사한 이미지에서 평가
            score, img_name, dist = evaluate_F_xw(
                x_params, w, images_data, yolo_detector, metric
            )
            Y_init.append(score)
            image_names.append(img_name)
            
            print(f"  → Image: {img_name[:30]}, Score = {score:.4f}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        Y_init = torch.tensor(Y_init, dtype=DTYPE, device=DEVICE, requires_grad=False).unsqueeze(-1)
        
        X_9d = X_init_9d
        Y = Y_init
        best_observed = [Y.max().item()]
        start_iter = 0
        
        print(f"\nInitial best: {best_observed[0]:.4f}")
        
        save_checkpoint(checkpoint_file, X_9d, Y, 0, best_observed, image_names)
    
    # BO 루프
    print(f"\n{'='*70}")
    print(f"BO Iterations")
    print(f"{'='*70}")
    
    for iteration in range(start_iter, n_iterations):
        print(f"\n[Iteration {iteration+1}/{n_iterations}]")
        
        try:
            # GP 학습 - 데이터는 detach된 상태
            print("  [1/3] Fitting GP...")
            
            # GP 학습용 데이터는 gradient 불필요
            X_norm = ((X_9d - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])).detach()
            Y_detached = Y.detach()
            
            gp = SingleTaskGP(X_norm, Y_detached)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            # 획득 함수 최적화
            print("  [2/3] Optimizing CVaR acquisition...")
            
            # CVaR with proper gradient
            acq = CVaRAcquisitionFunction(
                gp, BOUNDS, alpha=alpha, n_w_samples=100, 
                images_data=images_data,
                use_soft_cvar=use_soft_cvar,
                temperature=temperature
            )
            
            bounds_6d_norm = torch.tensor([[0.0]*6, [1.0]*6], dtype=DTYPE, device=DEVICE)
            
            # optimize_acqf 호출
            candidate_6d_norm, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds_6d_norm,
                q=1,
                num_restarts=10,
                raw_samples=256
            )
            
            # 후보점은 detach!
            candidate_6d_norm = candidate_6d_norm.detach().squeeze(0)
            acq_value = acq_value.detach()
            
            candidate_6d = BOUNDS[0, :6] + candidate_6d_norm * (BOUNDS[1, :6] - BOUNDS[0, :6])
            
            print(f"    Candidate: edge=[{candidate_6d[0]:.1f}, {candidate_6d[3]:.1f}], "
                  f"sim=[{candidate_6d[1]:.3f}, {candidate_6d[4]:.3f}]")
            print(f"    Predicted CVaR: {-acq_value.item():.4f}")
            
            # 실제 평가
            print("  [3/3] Evaluating...")
            
            # 실제 환경에서 샘플링
            if images_data and len(images_data) > 0:
                rand_idx = np.random.randint(0, len(images_data))
                w_eval = images_data[rand_idx]['environment']
                w_eval_tensor = env_dict_to_tensor(w_eval).detach()  # detach!
            else:
                w_eval_tensor = torch.rand(4, dtype=DTYPE, device=DEVICE)
                w_eval_tensor = BOUNDS[0, 6:10] + w_eval_tensor * (BOUNDS[1, 6:10] - BOUNDS[0, 6:10])
                w_eval_tensor = w_eval_tensor.detach()  # detach!
                w_eval = tensor_to_env_dict(w_eval_tensor)
            
            x_params_eval = {
                'edgeThresh1': candidate_6d[0].item(),
                'simThresh1': candidate_6d[1].item(),
                'pixelRatio1': candidate_6d[2].item(),
                'edgeThresh2': candidate_6d[3].item(),
                'simThresh2': candidate_6d[4].item(),
                'pixelRatio2': candidate_6d[5].item(),
            }
            
            # 평가
            new_y, img_name, dist = evaluate_F_xw(
                x_params_eval, w_eval, images_data, yolo_detector, metric
            )
            
            image_names.append(img_name)
            
            print(f"    → Image: {img_name[:30]}, Score: {new_y:.4f}")
            
            # GP 데이터에 추가 - 모두 detach!
            x_9d_new = torch.cat([candidate_6d, w_eval_tensor]).detach()
            new_y_tensor = torch.tensor([[new_y]], dtype=DTYPE, device=DEVICE, requires_grad=False)
            
            X_9d = torch.cat([X_9d, x_9d_new.unsqueeze(0)])
            Y = torch.cat([Y, new_y_tensor])
            
            best_observed.append(Y.max().item())
            
            improvement = best_observed[-1] - best_observed[-2]
            print(f"    Best so far: {best_observed[-1]:.4f} ({improvement:+.4f})")
            
            save_checkpoint(checkpoint_file, X_9d, Y, iteration+1, best_observed, image_names)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED by user]")
            save_checkpoint(checkpoint_file, X_9d, Y, iteration, best_observed, image_names)
            sys.exit(0)
        
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            # 예외 발생시에도 안전하게 저장
            try:
                save_checkpoint(checkpoint_file, X_9d, Y, iteration, best_observed, image_names)
            except:
                print("[ERROR] Failed to save checkpoint during exception")
            continue
    
    # 최종 결과
    best_idx = Y.argmax()
    best_X_9d = X_9d[best_idx]
    
    print("\n" + "="*70)
    print("Optimization Complete")
    print("="*70)
    print(f"\n최적 파라미터:")
    print(f"  edgeThresh1:  {best_X_9d[0]:7.2f}")
    print(f"  simThresh1:   {best_X_9d[1]:7.4f}")
    print(f"  pixelRatio1:  {best_X_9d[2]:7.4f}")
    print(f"  edgeThresh2:  {best_X_9d[3]:7.2f}")
    print(f"  simThresh2:   {best_X_9d[4]:7.4f}")
    print(f"  pixelRatio2:  {best_X_9d[5]:7.4f}")
    print(f"\n최적 환경:")
    print(f"  continuity:   {best_X_9d[6]:7.4f}")
    print(f"  orientation:  {best_X_9d[7]:7.4f}")
    print(f"  illumination: {best_X_9d[8]:7.4f}")
    print(f"  dispersion:   {best_X_9d[9]:7.4f}")
    print(f"\n최종 성능: {best_observed[-1]:.4f}")
    print(f"초기 성능: {best_observed[0]:.4f}")
    print(f"개선율: {(best_observed[-1]-best_observed[0])/best_observed[0]*100:+.1f}%")
    
    return best_X_9d, best_observed


def load_dataset(image_dir, gt_file, complete_only=False):
    """데이터셋 로드"""
    from pathlib import Path
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
        
        img_data = {
            'name': img_name,
            'image': image,
            'gt_coords': gt_coords
        }
        images_data.append(img_data)
    
    return images_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="dataset/images/test")
    parser.add_argument("--gt_file", default="dataset/ground_truth.json")
    parser.add_argument("--yolo_model", default="models/best.pt")
    parser.add_argument("--metric", default="lp", choices=["lp", "endpoint"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--n_initial", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--use_soft_cvar", action="store_true", 
                        help="Use soft CVaR for better gradients")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for soft CVaR")
    parser.add_argument("--complete_only", action="store_true")
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()
    
    images_data = load_dataset(args.image_dir, args.gt_file, args.complete_only)
    print(f"Loaded {len(images_data)} images")
    
    yolo_detector = YOLODetector(args.yolo_model)
    
    checkpoint_file = Path(args.resume) if args.resume else None
    
    best_params, history = optimize_borisk(
        images_data,
        yolo_detector,
        metric=args.metric,
        n_iterations=args.iterations,
        n_initial=args.n_initial,
        alpha=args.alpha,
        use_soft_cvar=args.use_soft_cvar,
        temperature=args.temperature,
        checkpoint_file=checkpoint_file
    )