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

        # x별로 CVaR 계산 (N개 샘플을 n_w 단위로 묶음)
        n_samples = len(train_Y)
        n_groups = n_samples // self.n_w

        if n_groups == 0:
            return train_Y.max().item()

        best_cvar = float('-inf')
        n_worst = max(1, int(self.n_w * self.alpha))

        # 각 x에 대해 CVaR 계산
        for i in range(n_groups):
            start_idx = i * self.n_w
            end_idx = start_idx + self.n_w
            group_scores = train_Y[start_idx:end_idx]

            # CVaR 계산 (worst alpha%)
            worst_scores, _ = torch.topk(group_scores, n_worst, largest=False)
            cvar = worst_scores.mean().item()

            # 최대 CVaR 갱신 (maximize)
            if cvar > best_cvar:
                best_cvar = cvar

        return best_cvar
    
    def compute_kg_value_for_single_w(self, x_candidate, w_idx):
        """
        단일 (x, w) 쌍에 대한 Knowledge Gradient 계산

        BoRisk의 핵심: 각 (x, w) 쌍을 개별적으로 평가!

        Args:
            x_candidate: [9] 파라미터 후보
            w_idx: 환경 인덱스

        Returns:
            kg_value: KG 값 (Expected CVaR Improvement)
        """
        x_candidate = x_candidate.squeeze() if x_candidate.dim() > 1 else x_candidate

        # 1. 단일 (x, w) 쌍 생성
        w = self.w_set[w_idx]  # [6]
        xw_pair = torch.cat([x_candidate, w]).unsqueeze(0)  # [1, 15]

        # 2. 현재 GP의 posterior (단일 점)
        with torch.no_grad():
            posterior = self.gp.posterior(xw_pair)

        # 3. 판타지 샘플 생성 (미래 관측 시뮬레이션)
        fantasy_improvements = []

        for i in range(self.n_fantasies):
            # 판타지 관측 샘플링 (단일 값)
            fantasy_obs_raw = posterior.rsample()  # [1, 1, 1]
            fantasy_obs = fantasy_obs_raw.squeeze()  # scalar

            # 판타지 모델 생성 (1개 관측만 추가)
            fantasy_model = self._create_fantasy_model(xw_pair, fantasy_obs.unsqueeze(0))

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

        # new_Y를 1D로 보장 (이미 [n_w] shape이어야 함)
        if new_Y.dim() > 1:
            new_Y = new_Y.squeeze()

        # 새 데이터 추가
        updated_X = torch.cat([train_X, new_X])
        updated_Y = torch.cat([train_Y, new_Y])  # 둘 다 1D tensor

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
        획득 함수 최적화 - (x, w) 쌍 선택

        BoRisk의 핵심: 모든 (x, w) 조합을 평가하고 최고의 쌍 선택!

        Args:
            bounds: 파라미터 경계 [2, param_dim]
            n_candidates: x 후보 수

        Returns:
            best_x: 최적 파라미터 [1, param_dim]
            best_w_idx: 최적 환경 인덱스
            best_kg: KG 값
        """
        # 랜덤 x 후보 생성
        param_dim = bounds.shape[1]
        candidates = torch.rand(n_candidates, param_dim, dtype=self.dtype, device=self.device)
        candidates = bounds[0] + (bounds[1] - bounds[0]) * candidates

        # 모든 (x, w) 조합 평가
        best_kg = -float('inf')
        best_x = None
        best_w_idx = None

        print(f"  [BoRisk-KG] Evaluating {n_candidates} x candidates × {self.n_w} w environments = {n_candidates * self.n_w} combinations...")

        for i, x in enumerate(candidates):
            for w_idx in range(self.n_w):
                kg = self.compute_kg_value_for_single_w(x, w_idx)

                if kg > best_kg:
                    best_kg = kg
                    best_x = x
                    best_w_idx = w_idx

            # 진행 상황 출력 (10% 단위)
            if (i + 1) % max(1, n_candidates // 10) == 0:
                print(f"    Progress: {i+1}/{n_candidates} x candidates evaluated, best KG={best_kg:.6f}")

        print(f"  [BoRisk-KG] Best (x, w_idx={best_w_idx}): KG={best_kg:.6f}")

        return best_x.unsqueeze(0), best_w_idx, best_kg


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
        w_set: 환경 세트 [n_w, 6]
        bounds: 파라미터 경계
        alpha: CVaR threshold
        use_full_kg: True면 전체 KG, False면 간소화 버전

    Returns:
        best_x: 최적 파라미터 [1, param_dim]
        best_w_idx: 최적 환경 인덱스 (Full KG) 또는 랜덤 (Simplified)
        acq_value: 획득 함수 값
        method: 사용된 방법
    """
    n_w = w_set.shape[0]

    if use_full_kg:
        # 전체 Knowledge Gradient (느리지만 정확)
        try:
            borisk_acq = BoRiskAcquisition(gp, w_set, alpha, n_fantasies=16)
            best_x, best_w_idx, acq_value = borisk_acq.optimize(bounds)
            return best_x, best_w_idx, acq_value, "BoRisk-KG"
        except Exception as e:
            print(f"Full KG failed: {e}, using simplified version")
            import traceback
            traceback.print_exc()
            use_full_kg = False

    if not use_full_kg:
        # 간소화 버전 (빠름) - w는 랜덤 선택
        try:
            simple_kg = SimplifiedBoRiskKG(gp, w_set, alpha)

            # 후보 생성 및 평가
            n_candidates = 256
            param_dim = bounds.shape[1]
            candidates = torch.rand(n_candidates, param_dim, dtype=bounds.dtype, device=bounds.device)
            candidates = bounds[0] + (bounds[1] - bounds[0]) * candidates

            acq_values = simple_kg.compute_acquisition_value(candidates, bounds)

            best_idx = torch.argmax(acq_values)
            best_x = candidates[best_idx].unsqueeze(0)
            acq_value = acq_values[best_idx].item()

            # Simplified는 w를 선택하지 않으므로 랜덤 선택
            best_w_idx = np.random.randint(0, n_w)
            print(f"  [Simplified] Random w_idx={best_w_idx} selected")

            return best_x, best_w_idx, acq_value, "Simplified-CVaR-KG"

        except Exception as e:
            print(f"Simplified KG failed: {e}")
            import traceback
            traceback.print_exc()
            # 최후의 폴백
            param_dim = bounds.shape[1]
            x = torch.rand(1, param_dim, dtype=bounds.dtype, device=bounds.device)
            x = bounds[0] + (bounds[1] - bounds[0]) * x
            w_idx = np.random.randint(0, n_w)
            return x, w_idx, 0.0, "Random-fallback"


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