# 다음 세션 TODO

**Date**: 2025-11-15
**Context**: Metric 분석 및 Upper Point 교점 방식 이해 완료
**Status**: Session 13 결과 분석 완료, Metric 개선 방안 준비됨

---

## 📊 현재 상황 요약

### 핵심 발견

1. **Metric 문제 확인됨**
   - Distance threshold = 40px (너무 작음!)
   - Direction sim = 1/(1+slope_diff) (비선형, 가파름)
   - 1개 실패 시 평균 급락 (부분 점수 부족)
   - **Score 범위**: 0.25~0.83 (넓은 분포, 변별력 있음)
   - **평균 Score**: 0.48 (적절한지 불명확)

2. **Upper Point 교점 방식 이해**
   - ✅ AirLine이 ROI 내부 모든 선 저장 (`airline_lines_in_roi`)
   - ✅ RANSAC은 대표 세로선 1개 선택
   - ✅ 코드가 수직 교차 선(상단 가로선) 찾기
   - ✅ Longi 선과 교점 계산 → Upper Point
   - ⚠️ Fallback: ROI 경계까지 연장
   - **추정 성공률**: 85~90% (완벽하지 않음)

3. **Session 13 실험 결과 (137 iterations)**
   - CVaR: 0.47~0.57 (평균 0.48, 좁은 범위)
   - Score: 0.25~0.83 (평균 0.48, 넓은 범위)
   - **CVaR-Score correlation = 0.006** (거의 0!)
   - 검출 실패 의심: 9개 (6.6%, Score < 0.3)
   - 검출 성공: 8개 (5.8%, Score > 0.7)
   - **문제**: iteration JSON에 `detected_coords` 없음

---

## 🎯 다음 세션 우선순위

### Priority 1: Metric 개선 및 실험 준비 (필수!)

#### Task 1.1: detected_coords 저장 추가
**목적**: 검출 성공률 직접 확인 가능하게

**작업**:
```python
# optimization.py의 로깅 부분 (대략 line 600-650)
# iteration 결과 저장 시 detected_coords 추가

iteration_result = {
    "iteration": iteration,
    "parameters": {...},
    "cvar": cvar,
    "score": score,
    "detected_coords": detected_coords,  # ← 추가!
    ...
}
```

**파일**: `optimization.py`
**예상 시간**: 10분

---

#### Task 1.2: Metric 선택 및 구현
**목적**: 시각적 품질과 일치하는 Metric 사용

**추천 순위**:
1. **Endpoint Metric** (distance_scale=50) ← 가장 추천!
   - 가장 직관적
   - 계산 간단
   - 해석 용이

2. **Angle + Exponential** (distance_scale=100)
   - 방향 + 거리 모두 반영
   - 부드러운 gradient

**작업**:
1. `optimization.py`에 새 metric 함수 추가
2. `line_equation_evaluation` 교체 또는 병행 사용
3. 기존 결과(Session 13)를 새 metric으로 재계산
4. 시각적 비교 (Best/Worst iteration)

**참고**: `METRIC_ANALYSIS_AND_PROPOSALS.md` 참조

**파일**: `optimization.py`
**예상 시간**: 1-2시간

---

#### Task 1.3: 대표 케이스 시각화
**목적**: Metric이 실제로 품질을 잘 반영하는지 확인

**작업**:
1. Session 13에서 대표 iteration 선택:
   - Best: Iter 64 (score=0.83, image_idx=88)
   - Median: Iter 7 (score=0.48)
   - Worst: Iter 125 (score=0.26, image_idx=1)

2. 각 케이스 재검출 및 시각화:
   - GT 좌표 (초록)
   - 검출 좌표 (빨강)
   - 각 선별 점수 표시

3. 육안 vs Metric 점수 비교

**스크립트**: 기존 `visualize_detection_v2.py` 수정
**예상 시간**: 30분

---

### Priority 2: 새로운 실험 시작

#### Task 2.1: 소규모 테스트 (10 iterations)
**목적**: 새 Metric으로 빠른 검증

**실험 설정**:
```bash
python optimization.py \
    --iterations 10 \
    --n_initial 5 \
    --alpha 0.3 \
    --env_file environment_top6.json
```

**확인 사항**:
- [ ] CVaR-Score correlation 개선 (현재 0.006 → 0.3+?)
- [ ] Score 범위 적절 (0~1)
- [ ] Best iteration의 시각적 품질

**예상 시간**: 30분~1시간

---

#### Task 2.2: 환경 제거 실험 (안전책)
**목적**: Baseline 확보

**실험 설정**:
```bash
python optimization.py \
    --no_environment \
    --iterations 50 \
    --n_initial 15 \
    --alpha 0.3
```

**기대 효과**:
- Session 11처럼 CVaR 0.65~0.70 달성
- 환경 상관 역설 회피
- 안정적인 baseline

**참고**: `PARADOX_ANALYSIS.md`

**예상 시간**: 2-3시간

---

### Priority 3: 추가 분석 (시간 있으면)

#### Task 3.1: Upper Point 검출 방식 통계
**목적**: 교점 vs Fallback 비율 확인

**방법**:
1. `full_pipeline.py`에 로깅 추가
2. Upper point 계산 시 어느 방식 사용했는지 저장
3. 통계 집계

**기대 결과**:
- 교점 성공: 85%
- Fallback: 15%

---

#### Task 3.2: 이미지 난이도 분류
**목적**: 어려운 이미지 vs 쉬운 이미지 분류

**방법**:
1. Session 13 결과에서 이미지별 평균 score 계산
2. 상위/하위 10개 이미지 분류
3. 시각적 특징 분석 (밝기, 칼라 유무 등)

**활용**:
- 어려운 이미지 제외하고 실험?
- 또는 환경 특징으로 반영?

---

## 📁 생성된 파일들

### 분석 문서
- ✅ `METRIC_ANALYSIS_AND_PROPOSALS.md` - Metric 문제점 및 개선안
  - 4가지 Metric 제안 (Exponential, Angle, Endpoint, Weighted)
  - 설계 철학 및 비교표
  - 추천: Endpoint (distance_scale=50)

- ✅ `analyze_experiment_coverage.py` - Session 13 실험 결과 분석
  - Score 통계: 평균 0.48, 범위 0.25~0.83
  - 검출 실패 의심: 9개 (6.6%)
  - 고유 이미지: 54개 / 137 evaluations

- ✅ `check_detection_coverage.py` - 전체 데이터셋 Coverage 확인 (미사용)

### 기존 문서
- `SESSION_13_FINAL_ANALYSIS.md` - 115 iterations 분석
- `PARADOX_ANALYSIS.md` - 환경 상관 역설

---

## 🔧 수정 필요 파일

### 1. `optimization.py`
**위치**: Line 600-650 (로깅 부분)
**수정**: `detected_coords` 저장 추가

**위치**: Line 52-147 (`line_equation_evaluation`)
**수정**: 새 Metric으로 교체 또는 추가

### 2. `visualize_detection_v2.py`
**수정**: 대표 케이스 시각화 기능 추가

---

## 💡 핵심 질문 (다음 세션 시작 시)

### Q1: Metric 어떤 것 사용할까?
- **추천**: Endpoint (distance_scale=50)
- **이유**: 가장 직관적, 계산 간단, 해석 용이
- **대안**: Angle + Exponential (더 이론적)

### Q2: 환경 특징 유지 vs 제거?
- **유지 (Top 6)**: 환경 효과 반영, 하지만 역설 위험
- **제거**: 안전한 baseline (CVaR 0.65~0.70 예상)
- **추천**: 제거 먼저 → 성공하면 유지 시도

### Q3: Upper Point 검출 개선 필요?
- **현재**: 85~90% 성공 추정
- **개선**: Fallback 비율 확인 후 결정
- **우선순위**: 낮음 (Metric이 먼저)

---

## ✅ 세션 시작 시 체크리스트

- [ ] `METRIC_ANALYSIS_AND_PROPOSALS.md` 읽기
- [ ] `analyze_experiment_coverage.py` 결과 확인
- [ ] Metric 선택 (Endpoint 추천)
- [ ] `optimization.py` 수정 (detected_coords + metric)
- [ ] 소규모 테스트 (10 iterations)
- [ ] 결과 확인 및 다음 단계 결정

---

## 📊 예상 타임라인

### 즉시 (1-2시간)
1. Metric 구현 및 테스트 (1시간)
2. 대표 케이스 시각화 (30분)
3. 소규모 실험 (30분)

### 단기 (오늘 저녁)
1. 환경 제거 실험 (2-3시간)
2. 결과 분석 및 비교
3. 다음 전략 결정

### 중기 (내일)
1. 최종 실험 (50 iterations)
2. 논문 Figure 생성
3. Session 13 vs 14 비교

---

## 🎓 배운 것

1. **Upper Point 교점 방식의 비밀**
   - AirLine이 모든 선 저장
   - 수직 교차 선 찾아서 교점 계산
   - Fallback은 ROI 경계 연장
   - → 매우 똑똑한 방법!

2. **Metric의 중요성**
   - BO의 목적함수가 잘못되면 최적화 불가능
   - 시각적 품질과 일치해야 함
   - 부드러운 gradient 필요 (BO friendly)

3. **환경 상관 역설**
   - 중간 강도(r=0.2~0.5)가 최악
   - 너무 약하면 무시 가능 (안전)
   - 너무 강하면 학습 가능 (데이터 많이 필요)

---

**마지막 업데이트**: 2025-11-15
**Status**: Metric 개선 준비 완료, 다음 실험 대기
**추천 첫 작업**: Task 1.2 (Endpoint Metric 구현)

**화이팅! 🚀**
