# 다음 세션 시작 가이드

**날짜**: 2025.11.11 19:35
**이전 세션**: Context 75% 사용으로 인한 세션 종료

---

## ⚡ 즉시 확인할 것

### 1. 테스트 결과 확인
```bash
# 백그라운드 프로세스 상태
ps aux | grep "python.*optimization.py"

# 로그 확인
tail -50 new_test.log

# 반복별 로그 확인
ls -lh logs/
cat logs/iter_001.json
cat logs/iter_002.json

# 최종 결과
ls -lh results/
cat results/bo_cvar_*.json | tail -1
```

### 2. CVaR 값 확인
- **이전 테스트**: CVaR = 0.0011 (매우 낮음)
- **기대값**: CVaR > 0.01 (평가 메트릭 개선으로)
- **성공 기준**: 이전 대비 10배 이상 개선

---

## 🔴 **최우선 작업: CVaR 계산 방식 수정**

### 문제점 (CRITICAL!)
현재 코드는 **BoRisk 알고리즘과 완전히 다름**:

**현재 잘못된 구현**:
```python
# optimization.py:217-273
def objective_function(X, images_data, yolo_detector, alpha=0.3):
    # 문제 1: 매번 모든 이미지(113개) 평가 → 매우 느림!
    scores = []
    for img_data in images_data:
        score = line_equation_evaluation(...)
        scores.append(score)

    # 문제 2: 직접 CVaR 계산 (GP 사용 안 함)
    cvar = np.mean(np.sort(scores)[:n_worst])
    return cvar
```

**BoRisk 올바른 방식** (BoTorch 튜토리얼 기반):
```python
# 1. 환경 변수 처리
# - 각 이미지 = 환경 w
# - w_set에서 n_w개 샘플링 (예: 10개)
w_set = sample_images(images_data, n_w=10)

# 2. GP 모델: (x, w) → y
model = SingleTaskGP(
    train_X,  # [N, 9+6] = params 9D + env 6D
    train_Y,
    input_transform=AppendFeatures(feature_set=w_set)
)

# 3. 획득 함수: qMultiFidelityKnowledgeGradient + CVaR
acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    num_fantasies=NUM_FANTASIES,
    objective=CVaR(alpha=0.3, n_w=n_w)
)

# 4. 매 iteration:
#    - 하나의 x 선택
#    - w_set의 몇 개 이미지만 평가 (10개, 113개 아님!)
#    - GP 업데이트
candidate = optimize_acqf(acqf, bounds)
observations = evaluate_on_w_samples(candidate, w_set)  # 10개만!
```

### BoRisk 논문에서 요구하는 방식
**GP를 활용한 CVaR 계산**:
```python
# TODO: optimization.py에 추가 필요
def compute_cvar_from_gp(gp, X, images_data, alpha=0.3, n_samples=1000):
    """
    GP 예측 분포에서 CVaR 계산

    Args:
        gp: 학습된 Gaussian Process
        X: 파라미터 [1, 9]
        images_data: 이미지 데이터 (환경 z 추출용)
        alpha: CVaR threshold (worst α%)
        n_samples: 몬테카를로 샘플 개수

    Returns:
        cvar: float
    """
    # 1. 각 이미지에 대해 환경 벡터 추출
    env_features = []
    for img_data in images_data:
        z = extract_environment(img_data['image'])  # 6D
        env_features.append(z)

    # 2. GP 입력: [x, z]
    X_with_env = []
    for z in env_features:
        x_z = torch.cat([X, torch.tensor([z])], dim=-1)  # [1, 15]
        X_with_env.append(x_z)

    X_batch = torch.cat(X_with_env, dim=0)  # [N_images, 15]

    # 3. GP로부터 예측 분포 샘플링
    with torch.no_grad():
        posterior = gp.posterior(X_batch)
        samples = posterior.rsample(torch.Size([n_samples]))  # [n_samples, N_images]

    # 4. 각 샘플에 대해 CVaR 계산
    cvars = []
    for i in range(n_samples):
        sample_scores = samples[i]  # [N_images]
        n_worst = max(1, int(len(sample_scores) * alpha))
        worst = torch.topk(sample_scores, n_worst, largest=False).values
        cvars.append(worst.mean().item())

    # 5. 평균 CVaR 반환
    return np.mean(cvars)
```

### 수정 계획 (BoRisk 정식 알고리즘)

#### Step 1: 환경 변수 추출 및 w_set 구성
```python
from environment_independent import extract_environment

# 모든 이미지의 환경 벡터 미리 추출
all_env_features = []
for img_data in images_data:
    env = extract_environment(img_data['image'])  # 6D
    all_env_features.append(torch.tensor([
        env['brightness'], env['contrast'], env['edge_density'],
        env['texture_complexity'], env['blur_level'], env['noise_level']
    ]))

# w_set: 매 iteration마다 n_w개 샘플링 (예: 10~20개)
def sample_w_set(all_env_features, n_w=15):
    indices = torch.randperm(len(all_env_features))[:n_w]
    return torch.stack([all_env_features[i] for i in indices])
```

#### Step 2: GP 모델 구조 변경
```python
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import AppendFeatures

# GP 입력: [N, 9] params only
# AppendFeatures가 자동으로 w_set 추가 → [N*n_w, 15]
model = SingleTaskGP(
    train_X,  # [N, 9] params만
    train_Y,  # [N*n_w, 1] 각 x마다 n_w개 환경 평가
    input_transform=AppendFeatures(feature_set=w_set)
)
```

#### Step 3: 획득 함수를 BoRisk로 변경
```python
from botorch.acquisition.multi_fidelity import qMultiFidelityKnowledgeGradient
from botorch.acquisition.objective import GenericMCObjective

def cvar_objective(samples, alpha=0.3):
    """CVaR 계산 objective"""
    # samples: [n_samples, n_w, 1]
    n_worst = max(1, int(samples.shape[1] * alpha))
    worst_samples = torch.topk(samples, n_worst, dim=1, largest=False).values
    return worst_samples.mean(dim=1)

# BoRisk 획득 함수
acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    num_fantasies=64,  # 판타지 샘플 개수
    objective=GenericMCObjective(cvar_objective),
    project=lambda X: X[..., :9]  # w 제거, x만 반환
)
```

#### Step 4: 평가 방식 변경
```python
# 기존 (잘못됨): 모든 113개 이미지 평가
# 새 방식: w_set의 n_w개만 평가

def evaluate_on_w_set(params, images_data, env_features, w_set_indices):
    """
    하나의 params에 대해 w_set 이미지만 평가

    Args:
        params: [9D] 파라미터
        images_data: 전체 이미지 데이터
        env_features: 전체 환경 벡터 리스트
        w_set_indices: w_set에 선택된 이미지 인덱스

    Returns:
        scores: [n_w, 1] 각 환경에서의 성능
    """
    scores = []
    for idx in w_set_indices:
        img_data = images_data[idx]
        score = evaluate_single(params, img_data)
        scores.append(score)

    return torch.tensor(scores).unsqueeze(-1)

# 매 iteration:
candidate = optimize_acqf(acqf, bounds, q=1)
observations = evaluate_on_w_set(candidate, images_data, all_env, w_indices)
# observations: [n_w, 1] - w_set 크기만큼만 평가!
```

#### Step 5: BO 루프 수정
```python
for iteration in range(n_iterations):
    # 1. w_set 샘플링 (매번 새로 또는 고정)
    w_set, w_indices = sample_w_set(all_env_features, n_w=15)

    # 2. GP 모델 생성/업데이트
    model = SingleTaskGP(
        train_X,  # [N, 9]
        train_Y,  # [N*n_w, 1]
        input_transform=AppendFeatures(feature_set=w_set)
    )
    fit_gpytorch_mll(mll)

    # 3. 획득 함수 생성
    acqf = qMultiFidelityKnowledgeGradient(...)

    # 4. 다음 평가 지점 선택
    candidate, _ = optimize_acqf(acqf, bounds, q=1)

    # 5. w_set에서만 평가 (15개만!)
    new_Y = evaluate_on_w_set(candidate, images_data, all_env, w_indices)

    # 6. 데이터 추가
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_Y])
```

### 핵심 차이점 요약

| 항목 | 현재 (잘못됨) | BoRisk (올바름) |
|------|--------------|----------------|
| **평가 개수** | 매번 113개 전체 | 매번 n_w개 (10~20개) |
| **GP 모델** | (x) → y | (x, w) → y |
| **획득 함수** | EI/UCB | ρKG (qMFKG) |
| **CVaR 계산** | 직접 평가 | GP 샘플링 |
| **속도** | 매우 느림 | 빠름 (1/10) |

---

## 📋 완료된 작업 요약

### 1. 평가 메트릭 변경 ✅
- **파일**: `optimization.py:39-116`
- **함수**: `line_equation_evaluation()`
- **방식**: 직선 방정식 Ax + By + C = 0 기반
- **평가 지표**:
  - 방향 유사도: 법선 벡터 내적 (60% 가중치)
  - 평행 거리: GT 중점에서 검출 직선까지 (40% 가중치)

### 2. RANSAC 가중치 최적화 ✅
- **차원 확장**: 6D → 9D
- **새 파라미터**:
  - `ransac_center_w`: [0.0, 1.0]
  - `ransac_length_w`: [0.0, 1.0]
  - `ransac_consensus_w`: [1, 10] (정수)
- **수정 위치**:
  - `optimization.py:33-36` - BOUNDS
  - `optimization.py:296` - Sobol dimension
  - `optimization.py:238-240` - params 딕셔너리
  - `optimization.py:539-541` - 결과 저장

### 3. 로깅 최적화 ✅
- **화면 출력**: 최소화 (토큰 절약)
- **파일 저장**: `logs/iter_XXX.json`
- **포함 내용**:
  - iteration, acq_function, acq_value
  - parameters (9D 전체)
  - cvar, cvar_normalized

---

## 🔧 남은 작업 우선순위

### Priority 1: CVaR 계산 방식 수정 (Critical)
- [ ] GP 기반 CVaR 계산 함수 구현
- [ ] objective_function을 evaluate_real로 분리
- [ ] 획득함수 평가 시 GP 기반 CVaR 사용
- [ ] 초기화 단계만 직접 평가 사용

### Priority 2: 환경 변수 통합 (Critical)
- [ ] 9D → 15D 확장 (params 9D + env 6D)
- [ ] `extract_environment()` 함수 통합
- [ ] GP 입력: (x, z) → y
- [ ] 새로운 이미지 z*에서 최적 x* 예측

### Priority 3: 판타지 관측 구현 (High)
- [ ] CVaR Knowledge Gradient 획득함수
- [ ] 환경 조건부 예측
- [ ] 시나리오 기반 평가

### Priority 4: 환경 특징 강화 (Medium)
- [ ] CLIP 기반 shadow/noise 탐지
- [ ] PSNR/SSIM 추가
- [ ] 6D → 9D 또는 10D 확장

---

## 📝 중요 참고사항

### 대전제 (절대 잊지 말 것)
1. **하드코딩으로 우회하지 말고 문제의 본질을 해결하라**
2. **임시 해결책 사용 시 반드시 TODO 주석을 남겨라**

### 현재 코드의 임시 해결책
```python
# optimization.py:217 - TODO: GP 기반 CVaR로 변경 필요
def objective_function(X, images_data, yolo_detector, alpha=0.3):
    # 현재: 직접 평가 (임시)
    # 필요: GP 샘플링 기반 CVaR
    pass
```

### 핵심 파일 위치
- **메인 로직**: `optimization.py` (551 lines)
- **파이프라인**: `full_pipeline.py` (YOLO + AirLine)
- **환경 추출**: `environment_independent.py` (6D 벡터)
- **평가 함수**: `optimization.py:39-116` (line_equation_evaluation)
- **작업 로그**: `TRACKING.md` (상세 진행 상황)

### 실행 명령
```bash
# 현재 테스트 (진행 중)
python optimization.py --iterations 2 --n_initial 3 --alpha 0.3

# 다음 실험 (성공 시)
python optimization.py --iterations 20 --n_initial 10 --alpha 0.3

# 전체 실험
python optimization.py --iterations 30 --n_initial 15 --alpha 0.2
```

---

## 🔍 디버깅 체크리스트

테스트 실패 시 확인:
- [ ] `logs/` 디렉토리에 iter_*.json 파일 생성되었는가?
- [ ] CVaR 값이 0이 아닌가? (0이면 평가 실패)
- [ ] RANSAC 파라미터가 제대로 전달되는가?
- [ ] 직선 방정식 평가가 올바른가?
- [ ] GP 학습이 실패하지 않았는가?

---

**다음 세션 시작 시 이 파일을 먼저 읽으세요!**
