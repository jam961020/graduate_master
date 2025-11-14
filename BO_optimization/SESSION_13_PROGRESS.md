# Session 13 - 진행상황 기록

**Date**: 2025-11-15
**Status**: 🔧 구현 개선 진행 중
**Goal**: BoRisk 알고리즘 개선 및 평가 메트릭 수정

---

## 📊 Session 13 결과 요약

### 최종 성능
- **Total iterations**: 115
- **Best CVaR**: 0.5654 (Iter 9)
- **Best Score**: 0.8329 (Iter 64)
- **문제**: Iter 9 이후 107회 정체 (93%)

### 핵심 발견
1. **환경 상관관계 역설 확인**:
   - 환경 상관 약함 (r=0.12) → CVaR 0.6886 ✓
   - 환경 상관 중간 (r=0.33) → CVaR 0.5654 ✗ (-19%)

2. **CVaR vs Score 불일치**:
   - Correlation: r = 0.0057 (거의 0)
   - GP의 CVaR 예측 부정확

3. **KG 예측 실패**:
   - KG correlation = -0.253 (음수!)
   - 잘못된 방향으로 탐색

---

## 🔍 검출 알고리즘 분석 (2025-11-15)

### 현재 구현 방식

#### 1. YOLO + AirLine + RANSAC
```
이미지
  ↓ YOLO
여러 ROI (fillet, longi, collar)
  ↓ 각 ROI마다
AirLine Q + AirLine QG
  ↓ RANSAC
선의 방정식 (Ax + By + C = 0)
  ↓ 교점 계산
12개 좌표 생성
```

#### 2. 좌표 생성 (calculate_final_coordinates)
- **longi_left_lower**: Fillet ∩ Left Longi
- **longi_right_lower**: Fillet ∩ Right Longi
- **longi_left_upper**: Left Longi 상단 (ROI 또는 교점)
- **longi_right_upper**: Right Longi 상단
- **collar_left_lower**: Collar 선 교점
- **collar_left_upper**: Collar 선 교점

#### 3. 평가 대상 (line_equation_evaluation)
**현재 (4개 선)**:
1. longi_left (세로) ✓
2. longi_right (세로) ✓
3. **fillet (가로)** ✓ ← 오늘 추가
4. collar_left (세로) ✓

### 발견된 문제점

#### 1. 가로선 1개 → 2개로 분리 필요
**현재**:
```
longi_left_lower ━━━━━━━━━━━━━━━━━━━━━━━━━━ longi_right_lower
                  (1개 fillet 가로선)
```

**개선 필요** (collar 있을 때):
```
longi_left_lower ━━━━━ collar_left_lower ━━━━━━━━━━━ longi_right_lower
   (짧은 가로선)            (긴 가로선 = fillet)
```

**이유**:
- 하단 3개 점: longi_left_lower, collar_left_lower, longi_right_lower
- 현재는 collar 무시하고 longi_left ↔ longi_right 직선으로 연결
- 실제로는 collar 경계에서 가로선이 2개로 나뉨

#### 2. Score가 낮은 이유
**관찰**:
- 시각적으로 거의 완벽한 검출
- 하지만 Score = 0.5 ~ 0.6 (낮음)

**의심되는 원인**:
1. **평가 함수 문제**:
   - 기울기 차이 + 거리 기반 평가
   - 가중치가 부적절할 수 있음

2. **GT 라벨링 부정확**:
   - 수동 라벨링 오차
   - 픽셀 단위 오차도 큰 패널티

3. **1개 가로선 평가**:
   - collar 무시 → 큰 오차 발생
   - 2개로 분리하면 개선될 수 있음

---

## 🔧 오늘 작업 (2025-11-15)

### 완료된 작업

1. ✅ **Session 13 전체 분석** (115 iterations)
   - CVaR/Score 추이 분석
   - KG correlation 계산
   - 시각화 생성 (session13_analysis.png)

2. ✅ **검출 결과 시각화 V1**
   - Best/Median/Worst iteration 시각화
   - 하지만 GT 선 제대로 안 그려짐

3. ✅ **검출 결과 시각화 V2** (개선)
   - GT 4개 선 제대로 그리기 (초록)
   - 검출 4개 선 그리기 (빨강)
   - results 폴더에 체계적으로 저장
   - metadata.json 생성

4. ✅ **필렛 가로선 평가 추가**
   - `optimization.py` 수정
   - line_definitions에 fillet 추가
   - 이제 4개 선 평가: longi_left, longi_right, fillet, collar_left

### 진행 중

5. 🔄 **진행상황 기록** (이 파일)

### 다음 작업 (우선순위)

6. **가로선 2개 분리 구현**:
   - fillet 가로선 (긴 쪽)
   - collar 가로선 (짧은 쪽)
   - GT 구조 업데이트 필요

7. **평가 함수 재검토**:
   - Score가 왜 낮은지 분석
   - 가중치 조정
   - 또는 새로운 메트릭 추가

8. **Warm Start 구현**:
   - Phase 1: No environment (20 iterations)
   - Phase 2: Top 4 environment (50 iterations)

---

## 📁 생성된 파일들

### 분석 문서
- `SESSION_13_FINAL_ANALYSIS.md` - 115 iterations 전체 분석
- `PARADOX_ANALYSIS.md` - 환경 상관관계 역설 분석
- `SESSION_13_CONCLUSION.md` - Opus 제안 전략
- `NEXT_SESSION.md` - 다음 세션 가이드
- `SESSION_13_PROGRESS.md` - 이 파일 (진행상황)

### 시각화 결과
```
logs/run_20251114_172045/
  └── session13_analysis.png  (CVaR/Score 추이 그래프)

results/run_20251114_172045_detection/
  ├── iter009_best_cvar0.5654/
  │   ├── full_result.png     (GT 초록, 검출 빨강)
  │   └── metadata.json
  ├── iter007_median_cvar0.4759/
  │   ├── full_result.png
  │   └── metadata.json
  └── iter060_worst_cvar0.4676/
      ├── full_result.png
      └── metadata.json
```

### 코드 수정
- `optimization.py`: 필렛 선 평가 추가
- `visualize_detection_v2.py`: 개선된 시각화 스크립트

---

## 💡 핵심 인사이트

### 1. 교점 계산의 정확성
**놀라운 발견**:
- RANSAC은 선의 방정식만 구함
- 하지만 **교점 계산**으로 정확한 끝점 생성
- 선이 정확하면 → 교점도 정확

**공식**:
```python
# 두 선의 교점
Line1: A₁x + B₁y + C₁ = 0
Line2: A₂x + B₂y + C₂ = 0

교점: x = (B₁C₂ - B₂C₁) / (A₁B₂ - A₂B₁)
     y = (A₂C₁ - A₁C₂) / (A₁B₂ - A₂B₁)
```

### 2. Collar Plate 검출
- 사각형 → 3개 선으로 검출
  - top (위 가로선)
  - bottom (아래 가로선)
  - outer_vertical (바깥 세로선)
- 4개 교점으로 collar 좌표 계산

### 3. 평가의 복잡성
- GT = 12개 점 (수동 라벨링)
- 검출 = 12개 점 (교점 계산)
- 평가 = 4~6개 선의 방정식 비교
  - 기울기 차이
  - 평행 거리

**문제**: 픽셀 단위 오차도 큰 패널티
**해결**: 상대적 메트릭 필요?

---

## 🎯 다음 단계 계획

### 우선순위 1: 가로선 2개 분리
**목표**: 더 정확한 평가

**작업**:
1. GT 구조 확인:
   - collar_left_lower가 하단 가로선상에 있는지?
   - 있으면 2개 가로선으로 분리 가능

2. 평가 함수 수정:
   - fillet_left: longi_left_lower ↔ collar_left_lower
   - fillet_right: collar_left_lower ↔ longi_right_lower
   - (또는 긴/짧은 쪽 자동 판단)

3. 시각화 업데이트:
   - 6개 선 모두 그리기

### 우선순위 2: 평가 메트릭 분석
**목표**: Score가 왜 낮은지 이해

**작업**:
1. Best iteration (Iter 9) 상세 분석:
   - 각 선별 score
   - 기울기 차이 / 거리 기여도
   - GT vs 검출 좌표 비교

2. 메트릭 개선:
   - 가중치 조정
   - 정규화 방식 변경
   - 또는 IoU 같은 다른 메트릭 추가

### 우선순위 3: Warm Start 구현
**목표**: Session 14 실험 준비

**작업**:
1. `environment_top4.json` 생성
2. `warm_start_phase()` 함수 구현
3. 2단계 최적화 통합

---

## 📊 현재 상태 요약

### 알고리즘
- ✅ YOLO + AirLine + RANSAC 파이프라인
- ✅ 교점 기반 좌표 생성
- ✅ 4개 선 평가 (longi 2개 + fillet + collar)
- ⚠️ 가로선 1개 → 2개 분리 필요
- ⚠️ Score가 시각적 품질보다 낮음

### 실험
- ✅ Session 13 완료 (115 iterations)
- ✅ 역설 확인 (환경 r=0.33 → 성능 하락)
- ⚠️ CVaR-Score 불일치 (r=0.006)
- ⚠️ KG 예측 실패 (r=-0.253)

### 다음 세션 준비
- ⏳ Warm Start 전략 구현 중
- ⏳ Top 4 환경 특징 선정
- ⏳ 평가 메트릭 개선

---

## 🔬 실험 계획 (Session 14)

### Option A: No Environment (안전)
```bash
python optimization.py \
    --no_environment \
    --iterations 50 \
    --n_initial 15 \
    --alpha 0.3
```
- 예상 CVaR: 0.65-0.70
- Session 11 성공 재현

### Option B: Warm Start + Top 4
```bash
# Phase 1: Warm start (20회)
warm_start_phase(n_initial=20)

# Phase 2: BO with Top 4 (50회)
bo_phase(env_file='environment_top4.json')
```
- 예상 CVaR: 0.60-0.68
- 외삽 문제 완화

### 결정
- **먼저 A 실행** → 베이스라인 확보
- **A 성공 시 B 시도** → 환경 효과 추가

---

## 📝 TODO List

### 즉시 (오늘)
- [x] Session 13 분석 완료
- [x] 검출 결과 시각화 V2
- [x] 필렛 선 평가 추가
- [ ] 진행상황 기록 ← 현재
- [ ] 가로선 2개 분리 구현
- [ ] 평가 함수 재검토

### 단기 (1-2일)
- [ ] environment_top4.json 생성
- [ ] Warm start 구현
- [ ] Session 14 실험 (No environment)
- [ ] Session 14 실험 (Warm start)

### 중기 (1주)
- [ ] 논문 Figure 생성
- [ ] 역설 분석 정리
- [ ] Limitation 섹션 작성

---

---

## 🔧 추가 작업 완료 (2025-11-15 02:30)

### A. 가로선 2개 분리 ✅
**완료**:
- GT 데이터 확인: collar_left_lower가 하단 가로선상에 있음 (Y좌표 차이 28-103px)
- 평가 함수 수정: **5개 선 평가**
  1. longi_left (세로)
  2. longi_right (세로)
  3. **fillet_left** (왼쪽 가로선)
  4. **fillet_right** (오른쪽 가로선)
  5. collar_left (세로)
- 시각화 업데이트: 5개 선 모두 그리기

### B. Best Iteration 상세 분석 ✅
**발견 (Iter 9, CVaR=0.5654, Score=0.5127)**:
```
LONGI_LEFT:    DETECTION FAILED ✗ (score = 0.0)
LONGI_RIGHT:   ✓ (거리 15px, 기울기 차이 0.28)
FILLET_LEFT:   ✗ (거리 361px - longi_left 없어서)
FILLET_RIGHT:  ✓ (거리 11px, 기울기 차이 0.03)
COLLAR_LEFT:   ✓ (거리 7px, 기울기 차이 0.57)
```

**Score가 낮은 이유**:
- **LONGI_LEFT 검출 실패**가 주범!
- FILLET_LEFT도 연쇄적으로 실패
- 5개 선 중 2개 실패 → 전체 score 0.5

**시각적으로는 거의 완벽**: longi_right, fillet_right, collar 모두 10-15px 오차

### 생성된 파일
```
results/run_20251114_172045_detection/
  ├── iter009_best_cvar0.5654/
  │   ├── full_result.png      (5개 선 시각화)
  │   └── metadata.json
  ├── iter004_median_cvar0.4759/
  └── iter060_worst_cvar0.4676/
```

---

## 🎯 다음 세션 TODO (우선순위)

### 즉시 (Session 14)
1. **C. Warm Start 구현**:
   - environment_top4.json 생성
   - warm_start_phase() 함수 (20 iterations, no env)
   - Session 14 실험 (No environment baseline)

2. **검출 개선** (선택):
   - LONGI_LEFT 검출 실패 원인 분석
   - 파라미터 튜닝 또는 필터링 개선

3. **AirLine Raw 시각화** (선택):
   - ROI별 Q/QG 결과 확인
   - 검출 실패 케이스 디버깅

### 단기
- Session 14 결과 분석
- 논문 Figure 생성
- 역설 분석 정리

---

**마지막 업데이트**: 2025-11-15 02:30
**완료**: A, B (가로선 2개 분리 + 상세 분석)
**다음**: C (Warm Start) → Session 14
**Status**: ✅ Ready for Session 14
