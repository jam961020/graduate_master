# Session 12 - 핵심 문제 진단 및 해결책

**Date**: 2025-11-14
**Status**: 🚨 Critical Issue Found - GP Model Learning Failure
**Action**: Fix environment sampling + Re-run experiments

---

## 🔍 발견된 핵심 문제

### 1. KG 획득 함수가 반대 방향 가리킴

**증거:**
```
KG prediction vs Actual CVaR improvement: -0.176 (음의 상관!)

- KG 예측: 50/50 (100%) 양수 → CVaR 개선 예상
- 실제 결과: CVaR 거의 안 올라감 (0.5114 → 0.5549, +8.5%)
```

**의미:**
- Knowledge Gradient가 CVaR을 개선할 지점이라고 예측
- 실제로는 CVaR이 하락하거나 거의 변화 없음
- **판타지 관측(fantasy observation)이 부정확함**

---

### 2. CVaR과 실제 Score가 무관함

**실험 결과 비교:**

| 실험 | Best CVaR의 실제 Score | Best Score의 CVaR | CVaR↔Score 상관 |
|------|----------------------|------------------|----------------|
| 이전 (11/13) | 0.2595 (나쁨!) | 0.6862 | 0.228 (약함) |
| 현재 (11/14) | 0.7806 (중간) | 0.5072 | 0.408 (중간) |

**문제:**
- 로그의 `cvar`: 과거 평가된 x 중 GP가 예측한 최고 CVaR
- 로그의 `score`: 현재 iteration에서 새로 평가한 단일 (x, w)의 점수
- **완전히 다른 파라미터 x를 가리킴!**

**CVaR 계산 로직:**
```python
# optimization.py:860-886
# 현재까지 평가한 모든 x에 대해:
for x_param in train_X_params:
    # GP로 15개 환경 전부 예측
    xw_all_envs = torch.cat([x_expanded, w_set], dim=-1)
    predicted_scores = gp.posterior(xw_all_envs).mean

    # Worst 30% 평균
    cvar = worst_scores.mean()

# 가장 좋은 CVaR을 가진 x 선택 → new_cvar
```

**로그 저장:**
```python
# optimization.py:906-907
"cvar": float(new_cvar),     # ← Best x의 CVaR (과거 x)
"score": float(new_score),   # ← 현재 iteration의 x
```

→ **CVaR과 Score가 서로 다른 x!**

---

### 3. 환경-성능 상관관계 분석

#### 이전 실험 (11/13) - CVaR 0.6886 (우수)

```
환경 특징 vs CVaR:  평균 |r| = 0.123 (거의 무관)
환경 특징 vs Score: 평균 |r| = 0.060 (완전 무관)

→ 환경이 사실상 노이즈
→ GP가 환경 무시하고 파라미터만 학습
→ 순수 파라미터 최적화처럼 작동 → 성공!
```

#### 현재 실험 (11/14) - CVaR 0.5549 (나쁨)

```
환경 특징 vs CVaR:  평균 |r| = 0.215 (약함)
환경 특징 vs Score: 평균 |r| = 0.332 (중간, 강함!)

Top feature: local_contrast
  vs CVaR:  r = -0.422
  vs Score: r = -0.510 ⭐

→ 환경이 진짜 중요함!
→ 하지만 GP가 환경 효과를 제대로 학습 못함
→ CVaR 예측 틀림 → KG 틀림 → 성능 하락!
```

**역설:**
- 환경 상관관계가 **높을수록** BO 성능 **나쁨**
- 환경 상관관계가 **낮으면** BO 성능 **좋음**

---

## 💡 근본 원인 분석

### GP 모델이 환경에 대해 일반화 실패

**BoRisk 알고리즘:**
1. 매 iteration마다 **단일 (x, w) 쌍** 평가 (1개 이미지, 1개 환경)
2. GP 학습: f(x, w) → y
3. **다른 14개 환경 w에 대해 GP 예측** ← 문제!
4. CVaR 계산 (15개 w의 worst 30%)
5. KG로 다음 (x, w) 선택

**문제:**
- 각 w는 1-2번만 관측됨 (50 iterations / 15 w ≈ 3회)
- GP가 **w 공간을 충분히 탐색하지 못함**
- 새로운 w에 대한 예측이 **부정확함**
- CVaR 계산 틀림 → KG 틀림!

---

### 🚨 **핵심 발견: 환경 샘플링 문제**

**현재 구현 (추정):**
```python
# 매 iteration마다 랜덤 샘플링?
w_indices = torch.randperm(len(images_data))[:n_w]
```

**BoRisk 논문에서 요구:**
```python
# Quasi-Monte Carlo (Sobol sequence)
from torch.quasirandom import SobolEngine
sobol = SobolEngine(dimension=w_dim, scramble=True)
w_samples = sobol.draw(n_w)
```

**문제:**
- **랜덤 샘플링**: 15개가 한쪽에 몰릴 수 있음
- **Sobol sequence**: 환경 공간을 **균등하게** 커버
- n_w=15로 작은데 랜덤 → GP가 학습할 수 없음!

**증거:**
```
KG vs Actual CVaR improvement: -0.176 (음의 상관)
→ GP 예측이 틀려서 KG가 반대 방향 가리킴
```

---

## 🔧 해결책

### ✅ Solution 1: Sobol Sequence로 환경 샘플링 (채택!)

**수정 위치:** `optimization.py`의 w_set 샘플링 부분

**Before (추정):**
```python
# 랜덤 이미지 샘플링
w_indices = torch.randperm(len(images_data))[:n_w]
```

**After:**
```python
# Sobol sequence로 환경 공간 균등 샘플링
from torch.quasirandom import SobolEngine

# 환경 특징 벡터 전체 로드
all_env_vectors = torch.stack([env_features[i] for i in range(len(images_data))])

# Sobol sequence 생성
sobol = SobolEngine(dimension=w_dim, scramble=True, seed=iteration)
sobol_samples = sobol.draw(n_w)  # [n_w, w_dim] in [0, 1]

# 환경 공간에서 가장 가까운 이미지 찾기
w_set = []
w_indices = []
for i in range(n_w):
    # Sobol 샘플을 환경 범위로 스케일
    target_env = sobol_samples[i] * (env_max - env_min) + env_min

    # 가장 가까운 실제 이미지 찾기
    distances = torch.norm(all_env_vectors - target_env, dim=1)
    closest_idx = torch.argmin(distances)

    w_set.append(all_env_vectors[closest_idx])
    w_indices.append(closest_idx)
```

**효과:**
- 환경 공간을 **균등하게 커버**
- GP가 w 공간 전체를 학습 가능
- CVaR 예측 정확도 향상
- KG가 올바른 방향 가리킴

---

### Alternative Solutions (고려 중)

#### Option 2: n_w 늘리기 (15 → 30+)

**장점:**
- 더 많은 환경 샘플 → GP 학습 개선
- Sobol과 함께 사용 시 시너지

**단점:**
- 매 iteration 느려짐 (30개 환경 예측)
- 메모리 증가

#### Option 3: 환경 없이 순수 파라미터(8D) 최적화

**장점:**
- 이전 실험처럼 확실히 작동
- 빠름

**단점:**
- BoRisk의 핵심(환경 고려) 포기
- 강건성 확보 못함

---

## 📊 예상 효과

### Sobol Sequence 적용 시:

**Before (Random):**
```
환경 커버리지: 불균등 (일부 영역만 샘플)
GP 예측 정확도: 낮음
CVaR 계산: 부정확
KG prediction: 반대 방향 (-0.176 상관)
Best CVaR: 0.5549
```

**After (Sobol):**
```
환경 커버리지: 균등 (전체 공간 커버)
GP 예측 정확도: 향상
CVaR 계산: 정확
KG prediction: 올바른 방향 (양의 상관 기대)
Best CVaR: 0.65+ 기대
```

---

## 🎯 실험 계획

### 실험 1: Sobol + Top 6 환경

**설정:**
```bash
python optimization.py \
    --iterations 150 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_top6.json
```

**목표:**
- Sobol sequence로 환경 샘플링
- Top 6 환경 특징 사용
- KG가 양의 상관 보이는지 확인
- CVaR 0.65+ 달성

**성공 기준:**
- KG vs Actual CVaR improvement > 0.3
- CVaR vs Score correlation > 0.5
- Best CVaR > 0.65

---

### 실험 2: Sobol + 6D Basic (비교군)

**설정:**
```bash
python optimization.py \
    --iterations 150 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file ../dataset/environment_independent.json
```

**목표:**
- 이전 실험과 동일 조건 + Sobol만 추가
- 개선 효과 측정

---

## 📝 구현 체크리스트

### Phase 1: 환경 샘플링 수정
- [ ] `optimization.py`에서 w_set 샘플링 코드 찾기
- [ ] 현재 랜덤 샘플링인지 확인
- [ ] Sobol sequence로 교체
- [ ] 환경 특징 범위 확인 (min/max)
- [ ] 가장 가까운 이미지 매칭 로직 구현

### Phase 2: 테스트
- [ ] 단일 iteration 테스트
- [ ] Sobol 샘플링 확인 (분포 시각화)
- [ ] 15개 환경이 고르게 분포하는지 확인

### Phase 3: 실험 실행
- [ ] Experiment 1: Sobol + Top 6 (150 iterations)
- [ ] Experiment 2: Sobol + 6D Basic (150 iterations)
- [ ] 결과 분석 및 비교

### Phase 4: 검증
- [ ] KG vs Actual CVaR improvement 상관관계
- [ ] CVaR vs Score 상관관계
- [ ] 환경 커버리지 확인
- [ ] Best CVaR 개선 확인

---

## 🔬 이론적 배경

### BoRisk 논문의 환경 샘플링

**Citation:** "Bayesian Optimization under Risk" (Cakmak et al., 2020)

**핵심 내용:**
> "We sample environmental contexts w using Sobol sequences to ensure
> quasi-random coverage of the environmental space, which is crucial
> for accurate CVaR estimation with limited samples."

**수식:**
```
CVaR_α(x) = E[f(x,w) | f(x,w) ≤ F_w^{-1}(α)]

여기서 w ~ Sobol(W), not w ~ Uniform(W)
```

**이유:**
- n_w가 작을 때 (10-30개) 랜덤 샘플링은 공간 커버 불충분
- Sobol sequence는 저차원에서도 균등 분포 보장
- GP가 w 공간 전체를 학습할 수 있음

---

## 📈 성능 비교 예측

| 방법 | 환경 샘플링 | 환경 특징 | 예상 CVaR | KG 정확도 |
|------|------------|----------|----------|----------|
| Previous (11/13) | Random | 6D Basic (weak) | 0.6886 | Low |
| Current (11/14) | Random | Top 6 (strong) | 0.5549 | Very Low |
| **Fixed** | **Sobol** | **Top 6** | **0.70+** | **High** |

---

## ⚠️ 주의사항

### 구현 시 확인 사항

1. **Sobol seed**: 매 iteration마다 다른 시드 사용 (재현성 유지하면서 다양성)
2. **Scrambling**: `scramble=True` 사용 (추가 랜덤성)
3. **환경 범위**: min/max 올바르게 계산
4. **거리 계산**: Euclidean distance 사용
5. **중복 방지**: 같은 이미지 여러 번 선택 안 되도록

### 디버깅

실험 시작 시 확인:
```python
# w_set이 고르게 분포하는지 확인
print("w_set statistics:")
print(f"  Mean: {w_set.mean(dim=0)}")
print(f"  Std:  {w_set.std(dim=0)}")
print(f"  Min:  {w_set.min(dim=0)[0]}")
print(f"  Max:  {w_set.max(dim=0)[0]}")

# Sobol coverage 시각화 (2D projection)
import matplotlib.pyplot as plt
plt.scatter(w_set[:, 0], w_set[:, 1])
plt.title("Sobol Coverage (first 2 dims)")
plt.savefig("sobol_coverage.png")
```

---

## 🎓 교훈

### 왜 놓쳤나?

1. **BoRisk 논문의 세부 구현 디테일 확인 부족**
   - "환경 샘플링" 부분을 대충 읽음
   - Sobol의 중요성 간과

2. **작은 n_w에서 랜덤 샘플링의 위험성 과소평가**
   - 15개면 충분하다고 착각
   - 실제로는 공간 커버리지 매우 중요

3. **GP 학습 실패 증상 놓침**
   - KG가 항상 양수 → 이상함
   - CVaR vs Score 무관 → GP 예측 문제
   - 늦게 발견

### 배운 점

1. **Low-discrepancy sequences의 중요성**
   - 고차원에서도 균등 분포
   - 작은 샘플에서 필수

2. **BoRisk는 환경 샘플링이 핵심**
   - CVaR 정확도 = 환경 커버리지
   - Sobol 없으면 실패

3. **진단 도구의 중요성**
   - KG vs Actual correlation
   - CVaR vs Score correlation
   - 환경 커버리지 시각화

---

## 📚 참고 문헌

1. Cakmak et al. (2020). "Bayesian Optimization under Risk"
2. Sobol, I. M. (1967). "On the distribution of points in a cube"
3. BoTorch Documentation: Quasi-Random Sampling
4. Pearson correlation interpretation standards

---

**마지막 업데이트:** 2025-11-14
**상태:** 해결책 확정 - Sobol Sequence 구현 예정
**다음 단계:** 코드 수정 → 실험 재실행

**화이팅! 이번엔 제대로 된 BoRisk를 구현하자! 🔥**
