"""
BoRisk - Shape 디버깅 버전
획득 함수의 모든 shape를 출력하여 문제 파악
"""
import torch
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
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 0.0, 0.0],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 1.0, 1.0]
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
    print("\n[Precomputing environments...]")
    airline_config = {'edgeThresh': 0}
    
    for img_data in images_data:
        env = estimate_environment(img_data['image'], airline_config)
        img_data['environment'] = env


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


# ==================== 디버그용 CVaR 획득 함수 ====================

class DebugCVaRAcquisitionFunction(AcquisitionFunction):
    """
    Shape 디버깅을 위한 CVaR 획득 함수
    모든 중간 shape를 출력
    """
    
    def __init__(self, model, bounds, alpha=0.3, n_w_samples=100, images_data=None):
        super().__init__(model=model)
        self.bounds = bounds
        self.alpha = alpha
        self.n_w_samples = n_w_samples
        self.call_count = 0  # forward 호출 횟수 추적
        
        if images_data is not None:
            self.real_envs = self._extract_env_distribution(images_data)
            print(f"    [DEBUG] Real envs shape: {self.real_envs.shape if self.real_envs is not None else None}")
        else:
            self.real_envs = None
    
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
        X: 입력 파라미터
        Returns: CVaR 값
        """
        self.call_count += 1
        
        # 첫 몇 번의 호출만 디버깅
        debug = (self.call_count <= 5)
        
        if debug:
            print(f"\n    [DEBUG Forward Call #{self.call_count}]")
            print(f"      Input X shape: {X.shape}")
            print(f"      Input X dim: {X.dim()}")
            print(f"      Input X dtype: {X.dtype}")
            print(f"      Input X device: {X.device}")
        
        # Shape 처리
        if X.dim() == 1:
            X = X.unsqueeze(0)
            if debug:
                print(f"      After unsqueeze: {X.shape}")
        
        original_shape = X.shape[:-1]
        X_flat = X.reshape(-1, 6) if X.shape[-1] == 6 else X.reshape(-1, X.shape[-1])
        batch_size = X_flat.shape[0]
        
        if debug:
            print(f"      Original shape: {original_shape}")
            print(f"      X_flat shape: {X_flat.shape}")
            print(f"      Batch size: {batch_size}")
        
        # CVaR 계산
        cvar_values = torch.zeros(batch_size, dtype=DTYPE, device=DEVICE)
        
        for i in range(min(batch_size, 2 if debug else batch_size)):  # 디버그 시 처음 2개만
            x_norm = X_flat[i:i+1]
            
            if debug and i == 0:
                print(f"      x_norm shape: {x_norm.shape}")
            
            # W 샘플링
            if self.real_envs is not None and len(self.real_envs) > 0:
                indices = torch.randint(0, len(self.real_envs), (self.n_w_samples,))
                w_samples = self.real_envs[indices]
                w_samples = (w_samples - self.bounds[0, 6:10]) / (self.bounds[1, 6:10] - self.bounds[0, 6:10])
            else:
                w_samples = torch.rand(self.n_w_samples, 4, dtype=DTYPE, device=DEVICE)
            
            if debug and i == 0:
                print(f"      w_samples shape: {w_samples.shape}")
            
            # X가 6차원이 아닌 경우 처리
            if x_norm.shape[-1] != 6:
                if x_norm.shape[-1] == 10:
                    # 이미 10차원인 경우, 처음 6차원만 사용
                    x_norm = x_norm[:, :6]
                    if debug:
                        print(f"      x_norm trimmed to 6D: {x_norm.shape}")
                else:
                    if debug:
                        print(f"      WARNING: Unexpected x_norm dimension: {x_norm.shape[-1]}")
            
            # [x, w] 결합
            x_repeated = x_norm.expand(self.n_w_samples, -1)
            xw = torch.cat([x_repeated, w_samples], dim=1)
            
            if debug and i == 0:
                print(f"      x_repeated shape: {x_repeated.shape}")
                print(f"      xw shape: {xw.shape}")
            
            # GP 예측
            with torch.no_grad():
                posterior = self.model.posterior(xw)
                mean = posterior.mean
                
                if debug and i == 0:
                    print(f"      posterior.mean shape: {mean.shape}")
                
                # Shape 처리
                if mean.dim() > 1:
                    mean = mean.squeeze(-1)
                    if debug and i == 0:
                        print(f"      mean after squeeze: {mean.shape}")
            
            # CVaR
            n_worst = max(1, int(self.n_w_samples * self.alpha))
            worst_vals = torch.topk(mean, n_worst, largest=False)[0]
            cvar = worst_vals.mean()
            
            cvar_values[i] = -cvar  # 음수로 반환 (maximize)
            
            if debug and i == 0:
                print(f"      CVaR value: {cvar_values[i]:.4f}")
        
        # 디버그 모드가 아닌 경우 나머지 배치 처리
        if not debug:
            for i in range(2, batch_size):
                x_norm = X_flat[i:i+1]
                
                if self.real_envs is not None and len(self.real_envs) > 0:
                    indices = torch.randint(0, len(self.real_envs), (self.n_w_samples,))
                    w_samples = self.real_envs[indices]
                    w_samples = (w_samples - self.bounds[0, 6:10]) / (self.bounds[1, 6:10] - self.bounds[0, 6:10])
                else:
                    w_samples = torch.rand(self.n_w_samples, 4, dtype=DTYPE, device=DEVICE)
                
                if x_norm.shape[-1] != 6:
                    if x_norm.shape[-1] == 10:
                        x_norm = x_norm[:, :6]
                
                x_repeated = x_norm.expand(self.n_w_samples, -1)
                xw = torch.cat([x_repeated, w_samples], dim=1)
                
                with torch.no_grad():
                    posterior = self.model.posterior(xw)
                    mean = posterior.mean.squeeze(-1) if posterior.mean.dim() > 1 else posterior.mean
                
                n_worst = max(1, int(self.n_w_samples * self.alpha))
                worst_vals = torch.topk(mean, n_worst, largest=False)[0]
                cvar_values[i] = -worst_vals.mean()
        
        # Shape 복원
        result = cvar_values.view(original_shape)
        
        if debug:
            print(f"      Result shape: {result.shape}")
            print(f"      Result dim: {result.dim()}")
            if result.numel() <= 5:
                print(f"      Result values: {result}")
        
        return result


# ==================== 간단한 테스트 함수 ====================

def test_acquisition_function():
    """획득 함수를 단독으로 테스트"""
    print("\n" + "="*70)
    print("Testing Acquisition Function Shapes")
    print("="*70)
    
    # 더미 데이터 생성
    n_data = 10
    X_dummy = torch.rand(n_data, 10, dtype=DTYPE, device=DEVICE)
    Y_dummy = torch.rand(n_data, 1, dtype=DTYPE, device=DEVICE)
    
    # GP 모델 생성
    X_norm = (X_dummy - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    gp = SingleTaskGP(X_norm, Y_dummy)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # 획득 함수 생성
    acq = DebugCVaRAcquisitionFunction(gp, BOUNDS, alpha=0.3, n_w_samples=10)
    
    # 다양한 입력 shape 테스트
    test_inputs = [
        torch.rand(6, dtype=DTYPE, device=DEVICE),  # 1D
        torch.rand(1, 6, dtype=DTYPE, device=DEVICE),  # 2D [1, 6]
        torch.rand(5, 6, dtype=DTYPE, device=DEVICE),  # 2D [5, 6]
        torch.rand(2, 3, 6, dtype=DTYPE, device=DEVICE),  # 3D [2, 3, 6]
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"\n  Test {i+1}: Input shape = {test_input.shape}")
        try:
            output = acq(test_input)
            print(f"    Success! Output shape = {output.shape}")
        except Exception as e:
            print(f"    Error: {e}")
    
    # optimize_acqf 테스트
    print("\n\n" + "="*70)
    print("Testing with optimize_acqf")
    print("="*70)
    
    bounds_6d_norm = torch.tensor([[0.0]*6, [1.0]*6], dtype=DTYPE, device=DEVICE)
    
    try:
        print("\n  Calling optimize_acqf...")
        candidate, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds_6d_norm,
            q=1,
            num_restarts=2,  # 작게 설정
            raw_samples=16  # 작게 설정
        )
        print(f"    Success!")
        print(f"    Candidate shape: {candidate.shape}")
        print(f"    Acq value shape: {acq_value.shape}")
        print(f"    Acq value: {acq_value.item():.4f}")
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== 메인 실행 ====================

def main():
    # 간단한 테스트만 실행
    test_acquisition_function()
    
    # 전체 파이프라인 테스트를 원하면 아래 주석 해제
    """
    print("\n\n" + "="*70)
    print("Loading Real Data for Full Test")
    print("="*70)
    
    # 데이터 로드
    from pathlib import Path
    images_data = load_dataset(
        image_dir="dataset/images/test",
        gt_file="dataset/ground_truth.json"
    )[:10]  # 10개만 테스트
    
    print(f"Loaded {len(images_data)} images")
    
    # 환경 계산
    precompute_all_environments(images_data)
    
    # YOLO 로드
    yolo_detector = YOLODetector("models/best.pt")
    
    # 최적화 실행 (짧게)
    optimize_borisk(
        images_data,
        yolo_detector,
        n_iterations=2,
        n_initial=5
    )
    """


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
    main()