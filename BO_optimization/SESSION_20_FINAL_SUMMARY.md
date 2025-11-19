# Session 20 최종 정리
**Date**: 2025-11-18
**Status**: 335장 30px threshold 실험 진행 중

---

## 🎯 오늘의 주요 작업

### 1. 335장 실험 분석 (20px threshold)

**결과** (run_20251118_055628):
```
Total iterations: 69/100
Best CVaR: 0.5394 (Iter 63)
Initial: 0.0084 → Final: 0.5053

문제 발견: CVaR 급락 (Iter 35-60)
- Iter 5-30:  0.47-0.50 (안정)
- Iter 35-60: 0.003-0.02 (급락!)
- Iter 60-69: 0.52-0.54 (회복)
```

**급락 원인 분석**:
```
Iter 35 (급락):
- CVaR: 0.0037
- Score: 0.8047 (선택된 이미지 1개는 좋음!)
- edgeThresh1: +4.894 (양수)

Iter 63 (Best):
- CVaR: 0.5394
- edgeThresh1: -21.787 (음수)

결론: 양수 edgeThresh 파라미터에서 대부분 이미지 검출 실패
→ 15개 환경 중 대부분 score=0
→ CVaR (worst 30%) ≈ 0
```

---

### 2. Quick Test 성공 (30장, 5 iters)

**결과** (run_20251118_053429):
```
Iter 1: CVaR=0.5682, score=0.7766
Iter 2: CVaR=0.5720  ↑
Iter 3: CVaR=0.5786  ↑
Iter 4: CVaR=0.5788  ↑
Iter 5: CVaR=0.5837  ↑ (Best!)

→ 지속적 개선 확인! ✓
→ Score=0: 0% (모두 성공!) ✓
```

---

### 3. GP(Gaussian Process) 이해

**핵심 개념**:
```
GP는 딥러닝과 다름!
- 딥러닝: gradient descent로 점진적 학습 (learning rate 있음)
- GP: closed-form solution으로 바로 posterior 계산 (learning rate 없음)

GP는 "함수의 확률 분포"를 모델링:
- 입력: (파라미터 x, 환경 w) = 14D
- 출력: score의 확률 분포 (mean, variance)
```

**중요**: CVaR은 GP 예측값이 아니라 **실제 평가 결과**!
```
매 iteration:
1. GP가 파라미터 x 선택 (acquisition function)
2. 이 x로 15개 이미지 전부 실제 평가
3. 15개 score의 worst 30% 평균 = CVaR
```

---

### 4. Threshold 30px로 변경

**변경 이유**:
```
20px에서 score=0 비율이 높음
→ 특정 파라미터에서 대부분 이미지 검출 실패
→ CVaR 급락 발생

30px로 관대하게:
- Score=0 비율 감소 예상
- 더 부드러운 gradient
- 안정적인 학습
```

**변경 위치** (optimization.py):
- Line 419: `threshold=30.0`
- Line 476: `threshold=30.0`
- Line 623: `threshold=30.0`

---

### 5. 새 실험 시작 (335장, 30px)

**설정**:
```
데이터: 335장 전체
Iterations: 100
Initial samples: 10
CVaR α: 0.3
n_w: 15
Threshold: 30px (연속 LP_r)
```

**상태**:
- 프로세스: PID 23112 (6.8GB 메모리)
- 로그: `overnight_335_30px_20251118.log`
- 현재: 환경 특징 추출 중 (on-the-fly)

---

## 📊 실험 결과 비교

| 실험 | 데이터 | Threshold | Best CVaR | 문제 |
|------|--------|-----------|-----------|------|
| Quick test | 30장 | 20px | 0.5837 | 없음 ✓ |
| 335장 | 335장 | 20px | 0.5394 | CVaR 급락 |
| **진행 중** | **335장** | **30px** | **?** | **확인 필요** |

---

## 🔍 핵심 발견사항

### 1. CVaR 급락의 진짜 원인
```
문제: GP 문제가 아님!
원인: BO의 exploration 과정에서 나쁜 파라미터 영역 탐색

양수 edgeThresh (+4.894):
- 일부 이미지: score=0.8047 (좋음)
- 대부분 이미지: score=0.0000 (검출 실패)
- CVaR (worst 30%) = 0.0037

음수 edgeThresh (-21.787):
- 대부분 이미지에서 안정적 검출
- CVaR = 0.5394 (Best)
```

### 2. Score와 CVaR의 차이
```
Score: 선택된 이미지 1개의 평가 결과
CVaR: 15개 이미지의 worst 30% 평균

Iter 35 예시:
- Score = 0.8047 (1개 이미지 좋음)
- CVaR = 0.0037 (나머지 14개 대부분 실패)
```

### 3. 30장 vs 335장 차이
```
30장: 검출 가능한 좋은 이미지들만
→ Score=0 비율 0%
→ CVaR 안정적 개선

335장: 어려운 이미지 포함
→ 특정 파라미터에서 대량 실패
→ CVaR 급락 발생
```

---

## 💡 다음 세션 작업

### Priority 1: 30px 실험 결과 확인
```bash
cd BO_optimization

# 1. 진행 상황 확인
python quick_monitor.py

# 2. CVaR 추이
tail -100 overnight_335_30px_20251118.log | grep "CVaR="

# 3. Visualization
python visualization_exploration.py logs/run_20251118_XXXXXX
```

### Priority 2: 결과 분석
```
확인 사항:
1. CVaR 급락이 줄어들었는가?
2. Score=0 비율이 감소했는가?
3. Best CVaR이 개선되었는가?
4. 지속적 개선 경향이 있는가?
```

### Priority 3: 추가 조치 (필요시)
```
만약 30px도 문제 발생:
1. 연속성 있는 지표로 변경 (사용자 제안)
2. 파라미터 범위 제한 (edgeThresh 양수 제거)
3. α 증가 (0.3 → 0.4)
4. 문제 이미지 필터링
```

---

## 📁 중요 파일 위치

```
BO_optimization/
├── SESSION_20_FINAL_SUMMARY.md      # 이 파일
├── SESSION_20_SUMMARY.md            # 초기 정리
├── SCORING_EXPLANATION.md           # LP_r 점수 계산 설명
│
├── optimization.py                  # ✅ threshold=30.0으로 변경됨
├── evaluation.py                    # LP_r 연속 점수 구현
├── visualization_exploration.py     # 9-panel visualization
├── quick_monitor.py                 # 실시간 모니터링
│
├── logs/
│   ├── run_20251118_055628/        # 335장 20px (69 iters, CVaR 급락)
│   └── run_20251118_053429/        # Quick test 30장 (5 iters, 성공!)
│
├── overnight_335_30px_20251118.log  # 🔄 현재 실험 로그
└── overnight_335_full_20251118.log  # 이전 20px 실험 로그
```

---

## 🎓 논문 작성 포인트

### 실험 결과
```
1. 30장 Quick test: 지속적 개선 확인 (+2.7%)
2. 335장 20px: CVaR 급락 현상 발견 (exploration)
3. 335장 30px: 진행 중 (더 안정적 예상)
```

### 기술적 기여
```
1. BO exploration 과정에서의 실패 분석
2. Threshold 조정의 영향 분석
3. 연속 LP_r 메트릭의 효과 검증
4. 데이터 품질 vs 양의 trade-off 분석
```

### 향후 작업
```
1. 30px 실험 결과 분석
2. 최적 threshold 결정
3. 연속성 메트릭 개선 (필요시)
4. 최종 실험 및 논문 작성
```

---

## 🔧 기술 참고사항

### GP vs 딥러닝
| 항목 | GP | 딥러닝 |
|------|-----|--------|
| 학습 방식 | Closed-form | Gradient descent |
| 학습률 | 없음 | 있음 (조절 가능) |
| 데이터 추가 | 바로 posterior 계산 | 재학습 필요 |
| 불확실성 | 분산으로 표현 | 별도 구현 필요 |

### CVaR 계산 (α=0.3)
```python
# 매 iteration
1. 선택된 파라미터 x로 15개 이미지 평가
2. scores = [s1, s2, ..., s15]
3. worst_5 = sorted(scores)[:5]  # worst 30% (약 4-5개)
4. cvar = mean(worst_5)
```

### Threshold 의미
```
threshold = 30px
이미지 해상도: 2448×3264
30px = 가로 1.23%, 세로 0.92%

거리별 점수:
- 0-15px:  0.5-1.0 (좋음)
- 15-30px: 0.0-0.5 (보통)
- 30px+:   0.0 (실패)
```

---

## ⚠️ 주의사항

### 1. 현재 실험 진행 중
```
프로세스: 23112 (6.8GB 메모리)
로그: overnight_335_30px_20251118.log
상태: 환경 특징 추출 중 → 곧 실험 시작
```

### 2. InputDataWarning
```
파라미터가 [0,1]로 정규화 안 됨
→ GP 학습 불안정 가능
→ 다음 세션에서 해결 권장
```

### 3. Score=0 문제
```
원인: AirLine이 선을 검출 못 함 (detected_lines=0)
해결: threshold 증가 (20→30) + 연속 메트릭
```

---

## 📞 빠른 참조

### 실험 모니터링
```bash
# 진행 상황
python quick_monitor.py

# 로그 실시간
tail -f overnight_335_30px_20251118.log

# 프로세스 확인
tasklist | grep python
```

### 실험 결과 분석
```bash
# Visualization
python visualization_exploration.py logs/run_YYYYMMDD_HHMMSS

# CVaR 추이
for f in logs/run_*/iter_*.json; do python -c "import json; d=json.load(open('$f')); print(f'Iter {d[\"iteration\"]}: CVaR={d[\"cvar\"]:.4f}')"; done
```

---

**작성 시각**: 2025-11-18 17:00
**다음 확인**: 30px 실험 완료 후 (예상 8-10시간)

**🎯 목표**: Threshold 30px에서 CVaR 급락 없이 안정적 개선!
**🌙 수고하셨습니다! 내일 좋은 결과 기대합니다!**
