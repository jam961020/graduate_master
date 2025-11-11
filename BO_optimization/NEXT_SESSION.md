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

### 문제점
현재 `optimization.py`의 CVaR 계산은 **직접 평가 방식**:
```python
# optimization.py:217-273
def objective_function(X, images_data, yolo_detector, alpha=0.3):
    scores = []
    for img_data in images_data:
        # 실제로 모든 이미지에 대해 평가 실행
        score = line_equation_evaluation(...)
        scores.append(score)

    # 직접 계산된 scores에서 CVaR
    n_worst = max(1, int(len(scores) * alpha))
    worst_scores = np.sort(scores)[:n_worst]
    cvar = np.mean(worst_scores)
    return cvar
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

### 수정 계획
1. `environment_independent.py`의 `extract_environment()` 함수 확인
2. GP 입력 차원 확장: 9D → 15D (params 9D + env 6D)
3. `objective_function` 분리:
   - `evaluate_real()`: 실제 평가 (초기화 및 학습용)
   - `compute_cvar_from_gp()`: GP 기반 CVaR (획득함수 평가용)
4. 획득함수를 CVaR-aware로 변경

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
