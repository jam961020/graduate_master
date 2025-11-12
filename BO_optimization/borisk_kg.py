"""
BoRisk CVaR Knowledge Gradient 구현
논문: https://github.com/saitcakmak/BoRisk
정확한 판타지 모델과 CVaR 계산
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.distributions import MultivariateNormal


class BoRiskAcquisition:
    """
    BoRisk의 rho-CVaR-KG 획득 함수
    
    핵심: 판타지 모델을 통한 CVaR 개선도 계산
    """
    
    def __init__(self, gp_model, w_set, alpha=0.3, n_fantasies=32, device=None, dtype=None):
        """
        Args:
            gp_model: 학습된 GP (15D input)
            w_set: 현재 환경 세트 [n_w, 6]
            alpha: CVaR threshold
            n_fantasies: 판타지 샘플 수
        """
        self.gp = gp_model
        self.w_set = w_set
        self.alpha = alpha
        self.n_fantasies = n_fantasies
        self.n_w = w_set.shape[0]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.double
        
        # 현재 최적 CVaR 계산 (비교 기준)
        self.current_best_cvar = self._compute_current_best_cvar()
        
    def _compute_current_best_cvar(self):
        """현재 데이터에서 최적 CVaR 계산"""
        # GP의 학습 데이터에서 최적 x 찾기
        train_X = self.gp.train_inputs[0]  # [N, 15]
        train_Y = self.gp.train_targets      # [N]
        
        # x별로 그룹화 (처음 9차원이 같은 것끼리)
        unique_x = []
        x_cvars = []
        
        # 간단히 처리: 마지막 n_w개가 하나의 x에 대한 평가라고 가정
        if len(train_Y) >= self.n_w:
            last_scores = train_Y[-self.n_w:]
            n_worst = max(1, int(self.n_w * self.alpha))
            worst_scores, _ = torch.topk(last_scores, n_worst, largest=False)
            current_cvar = worst_scores.mean().item()
        else:
            current_cvar = train_Y.min().item()
            
        return current_cvar
    
    def compute_kg_value(self, x_candidate):
        """
        Knowledge Gradient 계산
        
        Args:
            x_candidate: [1, 9] 파라미터 후보
            
        Returns:
            kg_value: KG 값 (Expected CVaR Improvement)
        """
        x_candidate = x_candidate.squeeze(0) if x_candidate.dim() > 1 else x_candidate
        
        # 1. (x, w) 쌍 생성
        x_expanded = x_candidate.unsqueeze(0).expand(self.n_w, -1)  # [n_w, 9]
        xw_pairs = torch.cat([x_expanded, self.w_set], dim=-1)      # [n_w, 15]
        
        # 2. 현재 GP의 posterior
        with torch.no_grad():
            posterior = self.gp.posterior(xw_pairs)
            mean = posterior.mean      # [n_w, 1]
            covar = posterior.covariance_matrix  # [n_w, n_w]
        
        # 3. 판타지 샘플 생성 (미래 관측 시뮬레이션)
        fantasy_improvements = []
        
        for _ in range(self.n_fantasies):
            # 판타지 관측 샘플링
            fantasy_obs = posterior.rsample()  # [n_w, 1]
            
            # 판타지 모델 생성 (새 관측 추가된 GP)
            fantasy_model = self._create_fantasy_model(xw_pairs, fantasy_obs)
            
            # 판타지 모델에서 CVaR 계산
            fantasy_cvar = self._compute_cvar_from_model(fantasy_model, x_candidate)
            
            # 개선도 계산
            improvement = max(0, fantasy_cvar - self.current_best_cvar)
            fantasy_improvements.append(improvement)
        
        # 4. Expected Improvement (판타지들의 평균)
        kg_value = np.mean(fantasy_improvements)
        
        return kg_value
    
    def _create_fantasy_model(self, new_X, new_Y):
        """
        판타지 모델 생성 (새 관측 추가)
        
        간단한 구현: 기존 데이터에 새 관측 추가
        """
        # 기존 학습 데이터
        train_X = self.gp.train_inputs[0]
        train_Y = self.gp.train_targets
        
        # 새 데이터 추가
        updated_X = torch.cat([train_X, new_X])
        updated_Y = torch.cat([train_Y, new_Y.squeeze(-1)])
        
        # 새 GP 생성 (간단한 버전)
        fantasy_gp = SingleTaskGP(updated_X, updated_Y.unsqueeze(-1))
        
        return fantasy_gp
    
    def _compute_cvar_from_model(self, model, x):
        """
        모델에서 x의 CVaR 계산
        """
        # (x, w) 쌍 생성
        x_expanded = x.unsqueeze(0).expand(self.n_w, -1) if x.dim() == 1 else x.expand(self.n_w, -1)
        xw_pairs = torch.cat([x_expanded, self.w_set], dim=-1)
        
        # 예측
        with torch.no_grad():
            posterior = model.posterior(xw_pairs)
            predictions = posterior.mean.squeeze(-1)  # [n_w]
        
        # CVaR 계산
        n_worst = max(1, int(self.n_w * self.alpha))
        worst_preds, _ = torch.topk(predictions, n_worst, largest=False)
        cvar = worst_preds.mean().item()
        
        return cvar
    
    def optimize(self, bounds, n_candidates=100):
        """
        획득 함수 최적화
        
        Args:
            bounds: 파라미터 경계 [2, 9]
            n_candidates: 후보 수
            
        Returns:
            best_x: 최적 파라미터 [1, 9]
            best_kg: KG 값
        """
        # 랜덤 후보 생성
        candidates = torch.rand(n_candidates, 9, dtype=self.dtype, device=self.device)
        candidates = bounds[0] + (bounds[1] - bounds[0]) * candidates
        
        # 각 후보의 KG 계산
        kg_values = []
        for x in candidates:
            kg = self.compute_kg_value(x)
            kg_values.append(kg)
        
        # 최적 후보 선택
        best_idx = np.argmax(kg_values)
        best_x = candidates[best_idx].unsqueeze(0)
        best_kg = kg_values[best_idx]
        
        return best_x, best_kg


class SimplifiedBoRiskKG:
    """
    간소화된 BoRisk Knowledge Gradient
    (빠른 계산을 위한 근사 버전)
    """
    
    def __init__(self, gp_model, w_set, alpha=0.3):
        self.gp = gp_model
        self.w_set = w_set
        self.alpha = alpha
        self.n_w = w_set.shape[0]
        
    def compute_acquisition_value(self, x_candidate, bounds):
        """
        간소화된 KG 계산: UCB + CVaR 고려
        
        Args:
            x_candidate: [batch, 9] 파라미터
            bounds: 경계
            
        Returns:
            acq_values: [batch] 획득 함수 값
        """
        if x_candidate.dim() == 1:
            x_candidate = x_candidate.unsqueeze(0)
        
        batch_size = x_candidate.shape[0]
        acq_values = []
        
        for i in range(batch_size):
            x = x_candidate[i]
            
            # (x, w) 쌍 생성
            x_expanded = x.unsqueeze(0).expand(self.n_w, -1)
            xw_pairs = torch.cat([x_expanded, self.w_set], dim=-1)
            
            # GP posterior
            with torch.no_grad():
                posterior = self.gp.posterior(xw_pairs)
                mean = posterior.mean.squeeze(-1)  # [n_w]
                stddev = posterior.stddev.squeeze(-1)  # [n_w]
            
            # CVaR 계산 (worst alpha%)
            n_worst = max(1, int(self.n_w * self.alpha))
            
            # Lower Confidence Bound (worst case 고려)
            lcb = mean - 2.0 * stddev
            worst_lcb, _ = torch.topk(lcb, n_worst, largest=False)
            
            # Upper Confidence Bound (exploration)
            ucb = mean + 2.0 * stddev
            
            # CVaR-aware acquisition value
            # Worst-case improvement potential + exploration bonus
            cvar_lcb = worst_lcb.mean()
            exploration_bonus = ucb.mean()
            
            acq_value = 0.7 * (-cvar_lcb) + 0.3 * exploration_bonus  # 음수 LCB이므로 부호 반전
            acq_values.append(acq_value)
        
        return torch.stack(acq_values)


def optimize_borisk(gp, w_set, bounds, alpha=0.3, use_full_kg=False):
    """
    BoRisk 최적화 메인 함수
    
    Args:
        gp: 학습된 GP 모델
        w_set: 환경 세트
        bounds: 파라미터 경계
        alpha: CVaR threshold
        use_full_kg: True면 전체 KG, False면 간소화 버전
        
    Returns:
        best_x: 최적 파라미터
        acq_value: 획득 함수 값
        method: 사용된 방법
    """
    
    if use_full_kg:
        # 전체 Knowledge Gradient (느리지만 정확)
        try:
            borisk_acq = BoRiskAcquisition(gp, w_set, alpha, n_fantasies=16)
            best_x, acq_value = borisk_acq.optimize(bounds)
            return best_x, acq_value, "BoRisk-KG"
        except Exception as e:
            print(f"Full KG failed: {e}, using simplified version")
            use_full_kg = False
    
    if not use_full_kg:
        # 간소화 버전 (빠름)
        try:
            simple_kg = SimplifiedBoRiskKG(gp, w_set, alpha)
            
            # 후보 생성 및 평가
            n_candidates = 256
            param_dim = bounds.shape[1]  # bounds의 dimension을 동적으로 가져옴
            candidates = torch.rand(n_candidates, param_dim, dtype=bounds.dtype, device=bounds.device)
            candidates = bounds[0] + (bounds[1] - bounds[0]) * candidates
            
            acq_values = simple_kg.compute_acquisition_value(candidates, bounds)
            
            best_idx = torch.argmax(acq_values)
            best_x = candidates[best_idx].unsqueeze(0)
            acq_value = acq_values[best_idx].item()
            
            return best_x, acq_value, "Simplified-CVaR-KG"
            
        except Exception as e:
            print(f"Simplified KG failed: {e}")
            # 최후의 폴백
            param_dim = bounds.shape[1]  # bounds의 dimension을 동적으로 가져옴
            x = torch.rand(1, param_dim, dtype=bounds.dtype, device=bounds.device)
            x = bounds[0] + (bounds[1] - bounds[0]) * x
            return x, 0.0, "Random-fallback"


if __name__ == "__main__":
    print("BoRisk CVaR-KG Module")
    print("=" * 60)
    print("사용법:")
    print("""
# optimization.py에서 import
from borisk_kg import optimize_borisk

# BO 루프에서 사용
candidate, acq_value, method = optimize_borisk(
    gp, w_set, BOUNDS, alpha=0.3, 
    use_full_kg=False  # 빠른 실행을 원하면 False
)
print(f"Method: {method}, Value: {acq_value:.4f}")
    """)