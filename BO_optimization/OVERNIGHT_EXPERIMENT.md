# Overnight Experiment (2025-11-15)

**시작 시각**: 2025-11-15 05:43
**PID**: 7417
**예상 소요**: 5-7시간

---

## 🚀 실험 설정

### Metric: lp (F1 score)
- **threshold=20px** (엄격)
- threshold=50px → 60% perfect score (너무 관대)
- **threshold=20px로 변별력 확보**

### BO 설정
- **iterations: 100** (큰 실험!)
- **n_initial: 5** (초기 줄임)
- alpha: 0.3
- n_w: 15
- env_file: environment_top6.json

### 근거
- 15 iters 테스트에서 **우상향 경향** 확인
- Initial: 0.8968 → Best: 0.9098 (+1.5%)
- 횟수 늘리면 더 개선될 가능성

---

## 📊 이전 테스트 결과 (lp threshold=50px)

**문제점:**
```
Score Distribution (15 iters):
  Perfect (>=0.99): 60.0%  ← 너무 많음!
  High (0.8-0.99):  40.0%
  CVaR-Score corr: -0.05   ← 여전히 음수
```

**변별력 부족 → threshold 줄임**

---

## 🎯 기대 효과 (threshold=20px)

**1. 변별력 증가**
- Perfect score 감소 (60% → 30%?)
- Score 분포 확대 (0.8~1.0 → 0.5~1.0?)
- 더 엄격한 기준

**2. Correlation 개선?**
- 변별력 증가 → CVaR-Score 상관관계 개선?
- 목표: corr > 0.3

**3. 100 iterations로 충분한 탐색**
- 우상향 경향 확인됨
- 많은 반복으로 local optima 탈출
- 예상 Best CVaR: 0.92+?

---

## 📋 확인 사항 (기상 후)

### 1. 실험 완료 확인
```bash
tail -100 logs/overnight_lp20_100iters.log
```

### 2. 결과 분석
```bash
cd BO_optimization
python visualize_session.py  # 새로운 run 분석
```

### 3. 주요 지표
- [ ] Best CVaR > 0.92?
- [ ] CVaR-Score correlation > 0.3?
- [ ] 수렴 여부 (마지막 20 iters 안정?)
- [ ] Perfect score 비율 감소?

---

## 🔄 다음 전략 (결과에 따라)

### Case 1: 성공 (correlation > 0.3, CVaR > 0.92)
→ **라벨링 증가** + 계속 실험
→ 환경 Top 2로 축소 시도

### Case 2: 부분 성공 (correlation 개선, but < 0.3)
→ **threshold 더 줄이기** (15px 또는 10px)
→ 또는 환경 Top 2로 차원 축소

### Case 3: 실패 (correlation 여전히 음수)
→ **라벨링 증가** (200개+)
→ 환경 특징 재설계
→ (최후) 환경 제거

---

## 📝 환경-성능 상관관계 (재확인)

**Session 13 분석 결과:**

| 환경 특징 | 상관계수 | 유의성 |
|----------|---------|--------|
| **local_contrast** | **r=-0.45** | p<0.001 *** |
| **gradient_strength** | **r=-0.30** | p<0.05 * |
| clip_rough | r=+0.27 | p<0.05 * |
| clip_smooth | r=+0.22 | ns |
| brightness | r=+0.04 | ns |
| edge_density | r=+0.04 | ns |

**→ 환경 특징은 유의미함! 제거 NO, 개선 YES**

---

## 💾 모니터링 명령어

### 진행 상황 확인
```bash
tail -f logs/overnight_lp20_100iters.log
```

### 현재 iteration 확인
```bash
grep -E "Iter [0-9]+/100" logs/overnight_lp20_100iters.log | tail -5
```

### CVaR progression
```bash
grep -E "Iter [0-9]+/100.*Best=" logs/overnight_lp20_100iters.log
```

### 프로세스 확인
```bash
ps aux | grep "PID: 7417"
```

---

## 📊 예상 결과 (낙관적)

```
Initial CVaR: 0.70-0.80  (threshold=20px로 더 낮아질 것)
Best CVaR:    0.92-0.95  (100 iters로 충분한 탐색)
Improvement:  +15-20%
Correlation:  0.3-0.5   (변별력 증가)

Score Distribution:
  Perfect (>=0.99): 20-30%
  High (0.8-0.99):  40-50%
  Mid (0.5-0.8):    20-30%
```

---

## 🎓 학습 내용

### lp Metric 이해
- F1 score = 2 × (Precision × Recall) / (Precision + Recall)
- Precision = TP / (TP + FP) = 검출 픽셀 중 맞는 비율
- Recall = TP / (TP + FN) = GT 픽셀 중 검출한 비율
- **threshold = TP 판정 기준 (픽셀 거리)**

### Threshold 효과
- **큰 threshold (50px)**: 관대, Perfect score 많음, 변별력 낮음
- **작은 threshold (20px)**: 엄격, 변별력 높음, Score 낮아짐
- **적절한 값**: 실험으로 찾기 (20px 시도 중)

### 우상향 경향의 의미
- BO가 작동하고 있다는 증거
- 더 많은 iteration으로 개선 가능성
- 100 iters면 충분할 것

---

**마지막 업데이트**: 2025-11-15 05:43
**상태**: 실험 진행 중 (PID: 7417)
**예상 완료**: 05:43 + 6시간 = 11:43 (오전)

**잘 자요! 내일 좋은 결과 기대합니다! 🌙✨**
