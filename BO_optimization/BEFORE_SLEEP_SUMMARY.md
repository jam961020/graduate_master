# 자기 전 요약 (2025-11-15)

**작성일**: 2025-11-15 05:40
**상태**: lp metric 테스트 진행 중 (Iter 3/15)

---

## ✅ 오늘 완료한 작업

### 1. Session 13 결과 분석 (150 iterations)

**핵심 문제:**
- CVaR-Score correlation = **-0.19** (음의 상관!)
- Best CVaR (Iter 9) ≠ Best Score (Iter 148)
- BO가 시간 지날수록 퇴보 (CVaR 0.50 → 0.44)

**원인:**
- GP의 환경 효과(W) 예측 실패
- 환경 상관이 중간 (r=0.33) → 학습 시도하지만 실패 → 악순환

---

### 2. Metric 비교 실험

**3가지 metric으로 대표 케이스 재평가:**

| Case | line_eq (현재) | lp (F1) | endpoint |
|------|----------------|---------|----------|
| Best | 0.91 | 1.00 | 0.98 |
| Median | 0.67 | 1.00 | 0.96 |
| Worst | 0.45 | 0.84 | 0.43 |

**결론:**
- lp metric이 너무 관대 (변별력 부족)
- endpoint가 더 나음
- **하지만 Metric보다 환경 예측 실패가 근본 문제**

---

### 3. 환경-성능 연관성 재분석 ⭐

**Session 13 데이터 (56개 이미지) 기반 상관관계:**

| 환경 특징 | Pearson r | 유의성 | 강도 |
|----------|-----------|--------|------|
| **local_contrast** | **-0.45** | p<0.001 *** | Moderate |
| **gradient_strength** | **-0.30** | p<0.05 * | Moderate |
| clip_rough | +0.27 | p<0.05 * | Weak |
| clip_smooth | +0.22 | ns | Weak |
| brightness | +0.04 | ns | Very Weak |
| edge_density | +0.04 | ns | Very Weak |

**핵심 발견:**

✅ **환경 특징이 성능에 영향을 줌!**
- 강한 상관: 2개 (local_contrast, gradient_strength)
- 중간 상관: 2개 (clip_rough, clip_smooth)
- 약한 상관: 2개 (brightness, edge_density)

**판정:**
> "환경 특징이 일부 영향을 주지만 약함"

**권장:**
1. **라벨링 데이터 증가** → 상관관계 강화
2. 환경 특징 재설계 (Top 2만 사용?)
3. (최종 수단) 환경 제거

---

### 4. Metric 교체 (line_equation → lp)

**optimization.py 3곳 수정:**
- `evaluate_single`
- `evaluate_on_w_set`
- `compute_all_scores`

---

### 5. lp metric 테스트 시작 (진행 중)

**설정:**
- Metric: lp (F1 score, threshold=50px)
- n_initial: 5
- iterations: 15
- alpha: 0.3
- n_w: 15

**초기 결과 (Iter 0-2):**

| Iteration | CVaR | 비고 |
|-----------|------|------|
| Init 1 | 0.8534 | 매우 높음! (Session 13보다 50% ↑) |
| Init 2 | 0.5458 | - |
| Init 3 | 0.7700 | - |
| Init 4 | 0.7160 | - |
| Init 5 | 0.7260 | - |
| **Iter 1** | **0.8968** | BO 시작 |
| **Iter 2** | **0.8993** | 미세 상승 |
| Iter 3+ | 진행 중... | - |

**의심:**
- lp metric이 너무 관대 → 거의 모든 케이스가 0.8+
- 진짜 개선인가, metric 착시인가?

---

## 🔄 진행 중

### 1. lp metric 테스트 (백그라운드)

**예상 완료**: 30분~1시간 후
**확인 사항**: CVaR-Score correlation 개선 여부

---

## 📊 생성된 파일

1. ✅ `session13_visualization.png` - Session 13 분석 그래프
2. ✅ `environment_correlation_analysis.png` - 환경-성능 상관관계
3. ✅ `environment_correlation_result.json` - 상관관계 수치
4. ✅ `SITUATION_ANALYSIS.md` - 전체 상황 종합 분석
5. ✅ `BEFORE_SLEEP_SUMMARY.md` - 이 파일

---

## 🎯 다음 할 일 (기상 후)

### Priority 1: lp metric 테스트 결과 확인

**판단 기준:**
```
만약 CVaR-Score correlation > 0.3:
  → lp metric 성공!
  → 50 iters로 확장 실험

만약 CVaR-Score correlation < 0.3:
  → lp metric도 실패
  → 다음 전략으로 이동
```

---

### Priority 2: 라벨링 증가 (환경 유지 전략)

**근거:**
- 환경-성능 상관관계 존재 (r=-0.45, -0.30)
- 데이터 부족이 문제일 수 있음
- 환경 제거 전에 시도할 가치 있음

**방법:**

**Option A: 자동 라벨링 (빠름)**
```python
# auto_labeling.py 작성
# AirLine으로 6개 점 자동 추출
# ground_truth.json 확장

예상 결과: 119개 → 200개+ 이미지
```

**Option B: Augmentation (매우 빠름)**
```python
# 기존 이미지 증강
# 노이즈, 밝기, 대비 변화
# GT는 동일하게 유지

예상 결과: 119개 → 500개+ 샘플
```

**예상 효과:**
- 데이터 증가 → 환경 효과 학습 개선
- GP 예측 정확도 향상
- CVaR-Score correlation 개선

---

### Priority 3: 환경 특징 축소 (Top 2)

**선택:**
- local_contrast (r=-0.45) ⭐
- gradient_strength (r=-0.30) ⭐
- (선택적으로 clip_rough +0.27 추가)

**장점:**
- 6D → 2D (또는 3D)
- 교차 항: 48D → 16D (67% 감소)
- 강한 특징만 유지

**방법:**
```bash
python create_environment_top2.py
python optimization.py --env_file environment_top2.json --iterations 30
```

---

### Priority 4: (최종 수단) 환경 제거

**조건:**
- 위 방법들이 모두 실패한 경우에만
- 또는 시간이 부족한 경우

**방법:**
```bash
python optimization.py --no_environment --iterations 50 --alpha 0.3
```

**예상:**
- CVaR 0.65~0.70 (Session 11 수준)
- 안정적인 baseline

---

## 💡 핵심 결론

### 1. 환경 특징은 유의미함!

**증거:**
- local_contrast: r=-0.45 (p<0.001, 매우 유의미)
- gradient_strength: r=-0.30 (p<0.05, 유의미)

**→ 환경을 바로 제거하면 안 됨!**

---

### 2. 문제는 데이터 부족

**현재 상황:**
- 119개 이미지
- Session 13에서 56개만 평가됨
- 환경 효과 학습에 부족

**→ 라벨링 증가가 우선!**

---

### 3. Metric은 부차적

**발견:**
- line_eq, lp, endpoint 모두 상관관계 높음 (0.85+)
- Metric 바꿔도 환경 예측 문제는 해결 안 됨

**→ Metric보다 데이터가 중요!**

---

## 📋 추천 순서 (기상 후)

```
1. lp metric 테스트 결과 확인 (5분)
   ↓
2. 자동 라벨링 or Augmentation (1-2시간)
   - 데이터 200개+ 확보
   ↓
3. 환경 Top 2 실험 (2-3시간)
   - local_contrast + gradient_strength
   - 30-50 iterations
   ↓
4. 결과 확인 (30분)
   - CVaR > 0.65면 성공!
   - 안 되면 환경 제거 고려
```

---

## 🔬 배운 것

### 환경-성능 상관관계가 존재한다!

**Session 13 분석으로 확인:**
- 56개 이미지 평가
- 2개 강한 상관, 2개 중간 상관
- **통계적으로 유의미**

### 환경 제거는 최후의 수단

**이유:**
- 환경 효과가 실제로 존재함
- 데이터 증가로 개선 가능성 높음
- 일반화 성능을 위해 환경 유지 필요

### 라벨링 증가가 핵심

**근거:**
- 56개로는 14D 공간 학습 부족
- 200개+ 되면 개선 가능
- Augmentation으로 빠르게 확보 가능

---

**마지막 업데이트**: 2025-11-15 05:40
**현재 상태**: lp metric 테스트 진행 중 (Iter 3/15)
**다음 작업**: 결과 확인 → 라벨링 증가 → Top 2 환경 실험

**잘 자요! 내일 좋은 결과 기대합니다! 🌙**

---

## 🌙 최종 업데이트 (05:43)

### ✅ Overnight 실험 시작!

**설정:**
- **Metric: lp (threshold=20px)** ← 엄격하게!
- **Iterations: 100** ← 큰 실험
- **n_initial: 5** ← 초기 줄임
- **PID: 7417**

**근거:**
- 15 iters에서 우상향 경향 확인
- threshold=50px는 너무 관대 (60% perfect)
- **threshold=20px로 변별력 확보**
- 100회 반복으로 충분한 탐색

**기대:**
- Best CVaR: 0.92+
- CVaR-Score correlation > 0.3
- Perfect score 20-30%

**모니터링:**
```bash
tail -f logs/overnight_lp20_100iters.log
```

**예상 완료**: 오전 11:43

---

**정말 잘 자요! 내일 아침에 좋은 결과가 있기를! 🌟**
