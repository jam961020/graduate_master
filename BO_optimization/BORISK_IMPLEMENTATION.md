# BoRisk 구현 가이드

**참고 자료**:
- 논문: "Bayesian Optimization of Risk Measures" (NeurIPS 2020)
- BoTorch 튜토리얼: https://botorch.org/docs/tutorials/risk_averse_bo_with_environmental_variables/
- BoRisk GitHub: https://github.com/saitcakmak/BoRisk (archived, 기능은 BoTorch로 통합)

---

## 핵심 개념

### 목적 함수
```
minimize ρ[F(x, W)]
```
- x: 결정 변수 (파라미터, 9D)
- W: 환경 변수 (이미지 특징, 6D)
- F(x, W): 성능 함수 (블랙박스)
- ρ: 리스크 측도 (CVaR)

### 왜 BoRisk인가?
일반 BO는 평균 성능을 최적화하지만, BoRisk는 **최악 케이스 성능**을 최적화합니다.
- 평균은 좋지만 특정 상황에서 실패하는 파라미터 회피
- CVaR(α=0.3): 최악 30% 케이스의 평균 성능 최적화

---

## 알고리즘 구조

### 1. 초기화

```python
# 1.1 환경 벡터 추출 (모든 이미지)
all_env_features = []
for img in images_data:
    env = extract_environment(img['image'])
    all_env_features.append(torch.tensor([
        env['brightness'], env['contrast'], env['edge_density'],
        env['texture_complexity'], env['blur_level'], env['noise_level']
    ]))

# 1.2 초기 샘플링
n_initial = 15
n_w = 15  # 각 x마다 평가할 환경 개수

train_X = []  # [n_initial, 9] params
train_Y = []  # [n_initial * n_w, 1] 성능

for i in range(n_initial):
    # Sobol 샘플링
    x = sample_params()  # [9D]

    # w_set 샘플링
    w_indices = random.sample(range(len(all_env_features)), n_w)

    # w_set에서만 평가
    for idx in w_indices:
        y = evaluate(x, images_data[idx])
        train_X.append(x)
        train_Y.append(y)
```

### 2. GP 모델 학습

```python
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import AppendFeatures

# w_set 구성
w_set = torch.stack([all_env_features[i] for i in w_indices])  # [n_w, 6]

# GP 모델
model = SingleTaskGP(
    train_X=torch.stack(train_X),  # [N, 9] params만
    train_Y=torch.tensor(train_Y).unsqueeze(-1),  # [N*n_w, 1]
    input_transform=AppendFeatures(feature_set=w_set)
    # AppendFeatures가 자동으로 (x, w) 쌍 생성 → [N*n_w, 15]
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

### 3. 획득 함수 (ρKG)

```python
from botorch.acquisition.multi_fidelity import qMultiFidelityKnowledgeGradient
from botorch.acquisition.objective import GenericMCObjective

# CVaR objective
def cvar_objective(samples, alpha=0.3):
    """
    samples: [n_fantasies, n_w, 1]
    각 fantasy에 대해 n_w개 환경 평가의 CVaR 계산
    """
    n_worst = max(1, int(samples.shape[1] * alpha))
    worst = torch.topk(samples, n_worst, dim=1, largest=False).values
    cvar = worst.mean(dim=1)  # [n_fantasies]
    return cvar

# ρKG 획득 함수
acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    num_fantasies=64,  # 판타지 샘플 개수
    current_value=best_cvar,  # 현재 최고 CVaR
    objective=GenericMCObjective(cvar_objective),
    project=lambda X: X[..., :9]  # 판타지에서 w 제거, x만 반환
)
```

**획득 함수 의미**:
- Knowledge Gradient: "이 점을 평가하면 최종 해의 품질이 얼마나 개선될까?"
- ρKG: "이 x를 평가하면 CVaR이 얼마나 개선될까?"
- 환경 w를 고려하여 가장 정보량이 많은 x 선택

### 4. 최적화 루프

```python
for iteration in range(n_iterations):
    # 4.1 w_set 샘플링 (고정 또는 매번 새로)
    w_indices = random.sample(range(len(all_env_features)), n_w)
    w_set = torch.stack([all_env_features[i] for i in w_indices])

    # 4.2 GP 모델 업데이트
    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=AppendFeatures(feature_set=w_set)
    )
    fit_gpytorch_mll(mll)

    # 4.3 획득 함수 생성
    acqf = qMultiFidelityKnowledgeGradient(...)

    # 4.4 다음 평가 지점 선택
    candidate, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,  # [2, 9] params bounds만
        q=1,
        num_restarts=20,
        raw_samples=512
    )

    # 4.5 w_set에서만 평가 (핵심!)
    new_observations = []
    for idx in w_indices:
        score = evaluate(candidate, images_data[idx])
        new_observations.append(score)

    # 4.6 데이터 추가
    # candidate를 n_w번 반복
    train_X = torch.cat([train_X, candidate.repeat(n_w, 1)])
    train_Y = torch.cat([train_Y, torch.tensor(new_observations).unsqueeze(-1)])
```

---

## 현재 코드와 비교

### 잘못된 구현 (현재)

```python
def objective_function(X, images_data):
    # 문제 1: 모든 이미지(113개) 평가
    scores = []
    for img in images_data:  # 113번 루프!
        score = evaluate(X, img)
        scores.append(score)

    # 문제 2: 직접 CVaR 계산
    n_worst = int(len(scores) * 0.3)
    cvar = np.mean(np.sort(scores)[:n_worst])
    return cvar

# 문제 3: EI 획득 함수 사용
acqf = qExpectedImprovement(model, best_f=Y.max())

# 결과: 매우 느림, BoRisk 알고리즘 아님
```

### 올바른 구현 (BoRisk)

```python
# 1. w_set만 평가 (15개)
def evaluate_on_w_set(x, images_data, w_indices):
    scores = []
    for idx in w_indices:  # 15번만 루프!
        score = evaluate(x, images_data[idx])
        scores.append(score)
    return torch.tensor(scores)

# 2. GP가 (x,w) → y 학습
model = SingleTaskGP(
    train_X,  # [N, 9]
    train_Y,  # [N*15, 1]
    input_transform=AppendFeatures(feature_set=w_set)
)

# 3. ρKG 획득 함수
acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    objective=CVaR_objective,
    num_fantasies=64
)

# 결과: 빠름, BoRisk 알고리즘
```

---

## 구현 체크리스트

### Phase 1: 환경 변수 통합
- [ ] `extract_environment()` 함수 확인 및 테스트
- [ ] 모든 이미지의 환경 벡터 미리 추출
- [ ] w_set 샘플링 함수 구현

### Phase 2: GP 모델 변경
- [ ] `AppendFeatures` 입력 변환 추가
- [ ] train_X를 params만 [N, 9]로 변경
- [ ] train_Y를 [N*n_w, 1]로 확장

### Phase 3: 평가 함수 수정
- [ ] `evaluate_on_w_set()` 함수 구현
- [ ] 기존 `objective_function` 제거
- [ ] 초기화 단계에서 w_set 평가 구현

### Phase 4: 획득 함수 변경
- [ ] CVaR objective 함수 구현
- [ ] `qMultiFidelityKnowledgeGradient` 적용
- [ ] EI/UCB 제거

### Phase 5: 테스트
- [ ] n_w=10으로 속도 테스트
- [ ] CVaR 수렴 확인
- [ ] 전체 이미지 대비 성능 비교

---

## 예상 개선 효과

| 항목 | 현재 | BoRisk 구현 후 |
|------|------|---------------|
| Iteration당 평가 개수 | 113개 | 15개 |
| Iteration당 시간 | ~5분 | ~40초 |
| 20 iteration 시간 | ~100분 | ~13분 |
| 알고리즘 | 일반 BO | BoRisk |
| CVaR 최적화 | ❌ | ✅ |

---

## 참고: w_set 크기 선택

- **n_w = 10**: 빠르지만 CVaR 추정 부정확
- **n_w = 15**: 균형 (권장)
- **n_w = 20**: 정확하지만 느림
- **n_w = 30**: 매우 정확하지만 너무 느림

BoTorch 튜토리얼에서는 n_w=10~15 권장.
