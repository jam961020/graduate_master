# 전체 상황 분석 (2025-11-15)

**작성일**: 2025-11-15
**현재 시각**: 05:25
**세션**: Session 13 완료 → 새 metric 테스트 중

---

## 📊 Session 13 결과 (150 iterations, line_equation metric)

### 핵심 지표

| 항목 | 값 | 비고 |
|------|-----|------|
| **CVaR-Score correlation** | **-0.1933** | 음의 상관! (심각) |
| **Acq-CVaR correlation** | **-0.1744** | KG도 실패 |
| Best CVaR | 0.5654 (Iter 9) | 초반에만 좋음 |
| Best Score | 0.8468 (Iter 148) | 후반에 나옴 |
| Initial CVaR | 0.4787 | - |
| Final CVaR | 0.4572 | **퇴보!** |

### 문제점

1. **CVaR ≠ Score (완전히 역상관)**
   - BO가 CVaR를 높이려 최적화
   - 하지만 CVaR 높아도 실제 Score는 낮음
   - Best CVaR iteration (9)과 Best Score iteration (148)이 완전히 다름

2. **Iter 9 이후 퇴보**
   ```
   Iter 9:   CVaR = 0.5654 (최고점)
   Iter 148: CVaR = 0.4016 (낮음)
   개선율:   -19.1% (하락!)
   ```

3. **환경 효과(W) 예측 실패**
   - GP가 환경에 따른 성능 변화를 제대로 학습 못함
   - CVaR 계산이 부정확 → KG가 잘못된 방향 제시
   - 환경 상관 r=0.33 (중간) → 학습하려 하지만 실패

4. **BO 수렴 실패**
   ```
   초반 (Iter 0-9):    평균 CVaR = 0.4990
   후반 (Iter 140-149): 평균 CVaR = 0.4354
   → 시간이 지날수록 오히려 나빠짐
   ```

---

## 🔍 Metric 비교 실험 (3가지 metric)

### 대표 케이스 3개 재평가

| Case | Iter | line_eq (현재) | lp (F1) | endpoint |
|------|------|----------------|---------|----------|
| Best | 64 | 0.9054 | **1.0000** | 0.9800 |
| Median | 115 | 0.6698 | **1.0000** | 0.9586 |
| Worst | 28 | 0.4450 | 0.8400 | 0.4253 |

### 발견사항

1. **lp metric이 너무 관대**
   - Best와 Median 둘 다 만점 (1.0)
   - 변별력 부족 → BO 최적화 어려움
   - threshold=50px가 너무 큼

2. **endpoint와 lp 거의 동일**
   - correlation = 0.9994
   - 둘 다 거리 기반

3. **line_eq가 상대적으로 엄격**
   - Best: 0.91, Median: 0.67, Worst: 0.45
   - 변별력 있음
   - 하지만 threshold=40px로 너무 작을 수 있음

4. **Metric 문제보다 더 큰 문제 존재**
   - 3가지 metric 모두 상관관계 높음 (0.85+)
   - Metric을 바꿔도 환경 예측 문제는 해결 안 됨
   - **본질적 문제: GP의 환경 효과 학습 실패**

---

## 🧪 현재 테스트 (lp metric, 진행 중)

### 설정

```
Metric: lp (F1 score, threshold=50px)
n_initial: 5
iterations: 15
alpha: 0.3
n_w: 15
env_file: environment_top6.json
```

### 초기 결과 (Iter 0-2)

| Iteration | CVaR | Score | 비고 |
|-----------|------|-------|------|
| Init 1 | 0.8534 | 0.9609 | 매우 높음! |
| Init 2 | 0.5458 | 0.8497 | - |
| Init 3 | 0.7700 | 0.9387 | - |
| Init 4 | 0.7160 | 0.9024 | - |
| Init 5 | 0.7260 | 0.9269 | - |
| **BO Iter 1** | **0.8968** | 1.0000 | 개선! |
| **BO Iter 2** | **0.8993** | 0.8320 | 계속 상승 |

### 관찰

1. **lp metric으로 점수가 전체적으로 상승**
   - Session 13 초기: CVaR 0.47~0.56
   - 현재 초기: CVaR 0.54~0.85
   - **평균 50% 이상 높음!**

2. **Score도 매우 높음**
   - 대부분 0.8~1.0 범위
   - lp metric의 관대함 확인

3. **BO가 작동하는 것처럼 보임**
   - Iter 1: 0.8968
   - Iter 2: 0.8993
   - 미세하지만 상승 중

4. **하지만 의심스러운 점**
   - lp metric이 너무 관대 → 변별력 부족
   - 거의 모든 케이스가 0.8+ 점수
   - **진짜 개선인가, 아니면 metric 착시인가?**

---

## 💡 근본 원인 분석

### 문제의 본질

**Metric 문제가 아니라 BoRisk 알고리즘 자체의 한계!**

1. **환경 차원이 너무 많음 (6D)**
   - 환경 특징: brightness, contrast, edge_density, texture, blur, noise
   - 환경 효과를 14D 공간 (8D params + 6D env)에서 학습
   - 교차 항 고려 시 48D 효과
   - **200개 샘플로 부족**

2. **환경 상관이 애매함 (r=0.33)**
   - 너무 약하면: GP가 무시 → 안전 (Session 11, r=0.12, CVaR 0.69)
   - 너무 강하면: GP가 학습 가능 → 성공 (이론상)
   - **중간 (0.2~0.5): GP가 학습 시도하지만 실패 → 최악!**
   - Session 13이 정확히 이 경우

3. **초기 샘플 분포 문제**
   - Initial 샘플이 나쁜 영역에 집중
   - GP가 나쁜 영역에서 환경 효과 학습
   - 좋은 영역에서는 데이터 없음 → 외삽 예측 실패
   - **Warm start로 해결 가능**

4. **CVaR 계산의 순환 논리**
   ```
   GP 예측 → CVaR 계산 → KG 최적화 → 새 샘플 → GP 업데이트
   ↑_______________________________________________|

   만약 GP 예측이 틀리면?
   → CVaR 틀림 → KG 잘못된 방향 → 나쁜 샘플 → GP 더 나빠짐
   → 악순환!
   ```

---

## 🎯 해결 전략 우선순위

### Strategy 1: 환경 제거 (가장 안전, 빠름)

**방법:**
```bash
python optimization.py --no_environment --iterations 50 --alpha 0.3
```

**근거:**
- Session 11에서 이미 성공 (CVaR 0.6886)
- 환경 없으면 8D만 최적화 → 단순함
- 환경 상관 역설 회피
- **성공 확률: 90%**

**단점:**
- 환경 효과 무시 → 일반화 부족
- 특정 이미지에만 좋을 수 있음

**예상 결과:**
- Best CVaR: 0.65~0.70
- 안정적인 수렴
- Baseline으로 유용

---

### Strategy 2: Warm Start + Top 4 환경 (이론적으로 최선)

**Phase 1: Warm Start (환경 없음)**
```bash
# 20개 initial로 8D 파라미터만 최적화
# 전체 이미지 평가 → 좋은 파라미터 영역 찾기
# 예상 CVaR: 0.62+
```

**Phase 2: BO with Environment (12D)**
```bash
# Phase 1에서 찾은 좋은 X 주변에서
# 환경 4D 추가하여 fine-tuning
# 예상 최종 CVaR: 0.70+
```

**근거:**
- 초기 샘플 분포 문제 해결
- 환경 차원 축소 (6D → 4D)
  - Top 4: local_contrast (-0.51), clip_rough (-0.45), brightness (-0.36), clip_smooth (+0.34)
  - 교차 항: 48D → 32D (33% 감소)
- Multi-fidelity BO 개념

**단점:**
- 구현 복잡함
- Phase 1이 20×113×5초 = 2시간 소요
- 시간 많이 걸림

**예상 결과:**
- Warm start CVaR: 0.62~0.68
- Final CVaR: 0.70~0.75
- **성공 확률: 60%** (구현 시간 고려)

---

### Strategy 3: Metric 개선 + 환경 제거 (절충안)

**현재 테스트가 이것!**

**1단계: lp metric 테스트 (진행 중)**
- 15 iterations로 빠른 검증
- CVaR-Score correlation 확인
- **예상: lp가 너무 관대 → 실패 가능성 높음**

**2단계: endpoint metric 시도 (대안)**
- threshold를 30px로 줄여서 변별력 확보
- 또는 distance_scale=50으로 exponential decay
- **endpoint가 lp보다 나을 수 있음**

**3단계: 환경 제거**
- Metric과 관계없이 환경이 문제라면
- 환경 제거가 최종 해결책

**예상 결과:**
- lp metric: 변별력 부족으로 개선 미미
- endpoint: 약간 나을 수 있음
- 최종적으로 환경 제거 필요

---

## 📋 즉시 할 일 (우선순위)

### Priority 0: 현재 테스트 완료 대기 (30분)

- lp metric 테스트 15 iters 완료
- CVaR progression 확인
- Score와 correlation 분석

**판단 기준:**
```
만약 CVaR-Score correlation > 0.3:
  → lp metric 성공! 50 iters로 확장

만약 CVaR-Score correlation < 0.3:
  → lp metric도 실패
  → 환경 제거 전략으로 전환
```

---

### Priority 1: 환경 제거 실험 (2-3시간)

**가장 안전하고 빠른 성공 방법**

```bash
python optimization.py \
  --no_environment \
  --iterations 50 \
  --n_initial 15 \
  --alpha 0.3
```

**기대 효과:**
- CVaR 0.65~0.70 달성
- 안정적인 baseline
- Session 11 재현

---

### Priority 2: (선택) Warm Start 구현 (4-7시간)

**환경 제거가 성공하면:**
- Warm Start 구현
- 환경 포함 최종 실험
- 최고 성능 달성 시도

**환경 제거가 실패하면:**
- 파라미터 범위 재검토
- 또는 다른 근본적 문제 존재

---

## 🔬 배운 교훈

### 1. Metric은 부차적 문제

- line_eq, lp, endpoint 모두 상관관계 높음 (0.85+)
- Metric 바꿔도 환경 예측 문제는 해결 안 됨
- **본질: GP의 환경 효과 학습 실패**

### 2. 환경 상관의 역설

```
환경 상관 약함 (r<0.2):  GP 무시 → 안전 ✓
환경 상관 중간 (0.2~0.5): GP 학습 시도 → 실패 → 최악 ✗
환경 상관 강함 (r>0.5):  GP 학습 성공 → 좋음 ✓ (데이터 많이 필요)
```

**Session 13이 정확히 중간 (r=0.33)!**

### 3. BoRisk의 한계

- 고차원 (14D+) 환경 효과 학습은 어려움
- 초기 샘플 분포가 매우 중요
- Warm start 없으면 악순환 가능

### 4. 실용적 해결책

- **환경 제거가 가장 안전**
- 복잡한 알고리즘보다 단순한 접근이 나을 수 있음
- Baseline부터 확보하고 점진적 개선

---

## 📊 최종 권장사항

### 즉시 (오늘 저녁)

1. ✅ **lp metric 테스트 완료 대기** (30분)
2. 🔴 **환경 제거 실험 시작** (2-3시간)
   - 가장 안전한 선택
   - Session 11 재현
   - Baseline 확보

### 단기 (내일)

3. 결과 분석 및 비교
   - Session 11 vs Session 13 vs 환경 제거
4. 논문 Figure 생성
5. 다음 전략 결정

### 중기 (필요시)

6. Warm Start 구현
7. 최종 실험
8. 논문 작성

---

**마지막 업데이트**: 2025-11-15 05:25
**상태**: lp metric 테스트 진행 중 (Iter 2/15)
**다음**: 테스트 완료 → 환경 제거 실험

**핵심 결론: Metric 문제보다 환경 예측 실패가 근본 원인. 환경 제거가 가장 현실적 해결책.**
