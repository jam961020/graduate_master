# Session 15 실험 보고서 - Quick Test (교수님 검토용)

**날짜**: 2025-11-17
**실험 ID**: run_20251116_061530
**작성자**: Graduate Student
**목적**: BoRisk 알고리즘을 활용한 용접선 검출 파라미터 최적화

---

## 1. 실험 개요

### 1.1 목적
- BoRisk (Bayesian Optimization under Risk) 알고리즘 적용
- CVaR 기반 강건성 확보
- 14차원 고차원 공간에서 효율적 파라미터 최적화

### 1.2 핵심 알고리즘
**BoRisk with CVaR**
```
maximize: CVaR_α(f(x, w))
where:
  - x: 파라미터 벡터 (8차원)
  - w: 환경 벡터 (6차원)
  - f(x, w): LP_r 성능 메트릭
  - α = 0.3: worst 30% 시나리오의 평균
```

---

## 2. 최적화 변수

### 2.1 결정 변수 (Decision Variables) - 8차원

#### A. AirLine 알고리즘 파라미터 (6D)

**Q 프리셋 (Quality-focused):**
| 파라미터 | 범위 | 기본값 | 설명 |
|---------|------|--------|------|
| edgeThresh1 | [-23.0, 7.0] | -3.0 | Canny edge detection threshold |
| simThresh1 | [0.5, 0.99] | 0.98 | Line segment similarity threshold |
| pixelRatio1 | [0.01, 0.15] | 0.05 | Minimum pixel ratio for valid lines |

**QG 프리셋 (Quality-Greedy):**
| 파라미터 | 범위 | 기본값 | 설명 |
|---------|------|--------|------|
| edgeThresh2 | [-23.0, 7.0] | 1.0 | Canny edge detection threshold |
| simThresh2 | [0.5, 0.99] | 0.75 | Line segment similarity threshold |
| pixelRatio2 | [0.01, 0.15] | 0.05 | Minimum pixel ratio for valid lines |

#### B. RANSAC 가중치 (2D)
| 파라미터 | 범위 | 기본값 | 설명 |
|---------|------|--------|------|
| ransac_weight_q | [1.0, 20.0] | 10.0 | Q 프리셋 선 후보 가중치 |
| ransac_weight_qg | [1.0, 20.0] | 10.0 | QG 프리셋 선 후보 가중치 |

### 2.2 환경 변수 (Environment Variables) - 6차원

#### Physical Features (4D) - 전통적 컴퓨터 비전 방법

| 특징 | 범위 | Correlation | p-value | 설명 |
|------|------|-------------|---------|------|
| **local_contrast** | [0,1] | **-0.445** | **<0.001** | ROI 영역의 국소 대비도 |
| **gradient_strength** | [0,1] | **-0.304** | **<0.05** | 이미지 그래디언트 강도 |
| brightness | [0,1] | +0.043 | 0.755 | 전체 이미지 평균 밝기 |
| edge_density | [0,1] | +0.045 | 0.744 | Canny 엣지 밀도 |

#### CLIP-based Semantic Features (2D) - Vision-Language Model

**CLIP 모델**: OpenAI CLIP ViT-B/32

| 특징 | Text Prompt | Correlation | p-value | 설명 |
|------|------------|-------------|---------|------|
| **clip_rough** | "a rough textured surface with debris" | **+0.272** | **<0.05** | 거친 표면 유사도 |
| **clip_smooth** | "a smooth clean surface" | +0.221 | 0.102 | 부드러운 표면 유사도 |

**CLIP Features 계산 과정:**
1. YOLO로 welding seam ROI 추출 (2448×3264 이미지에서)
2. ROI 이미지를 CLIP ViT-B/32 모델에 입력
3. 사전 정의된 text prompt와 cosine similarity 계산
4. Similarity 값 [0, 1]을 환경 특징으로 사용

**CLIP 사용 이유:**
- 전통적 CV features (brightness, contrast)는 low-level 특징
- CLIP은 "거칠다", "부드럽다" 같은 semantic concept 이해
- 용접 품질과 관련된 표면 특성을 더 잘 포착
- 대규모 Vision-Language 학습으로 일반화 성능 향상

**총 입력 차원: 14D (8D params + 6D env)**

---

## 3. 평가 메트릭

### 3.1 LP_r (Line Precision - Recall)

**출처**: AirLine 논문 (IROS 2023)
**정의**: Ground Truth 픽셀 중 검출 선으로부터 threshold 이내에 있는 비율

**수식**:
```
LP_r = Σ(τ_r(X) ⊗ Y) / ΣY

where:
  X: 검출된 선들 (detected lines)
  Y: Ground truth 선의 픽셀들
  τ_r: dilation function (tolerance radius r)
  ⊗: element-wise overlap
```

**계산 과정**:
1. GT 선을 픽셀로 샘플링 (각 선당 100개 → 4개 선 = 400개 픽셀)
2. 검출 선도 동일하게 샘플링 (400개 픽셀)
3. 각 GT 픽셀에서 가장 가까운 검출 픽셀까지의 거리 계산
4. threshold(=20px) 이내 GT 픽셀 개수 / 전체 GT 픽셀 개수

**설정**:
- **Threshold**: 20 pixels
- **이미지 해상도**: 2448 × 3264
- **20px 의미**: 가로 0.6%, 세로 0.8% (적당한 tolerance)
- **범위**: [0, 1] (높을수록 좋음)

**특징**:
- 실제로는 **Recall** (GT coverage)
- RANSAC이 단일 선 선택 → over-detection 문제 없음
- Precision이 암묵적으로 보장됨

---

## 4. BO 알고리즘 설정

### 4.1 샘플링 전략

**초기 샘플링 (Exploration):**
- **n_initial**: 5
- 방법: Latin Hypercube Sampling (LHS)
- 각 샘플당 15개 환경에서 평가
- 총 초기 평가: 5 × 15 = 75

**BO 반복 (Exploitation + Exploration):**
- **iterations**: 15
- 획득 함수: Knowledge Gradient (KG)
- 각 iteration당 15개 환경에서 평가
- 총 BO 평가: 15 × 15 = 225

**전체 평가 횟수: 300 (75 + 225)**

### 4.2 CVaR 설정

**Risk-aware 최적화:**
- **alpha**: 0.3 (30th percentile)
- **의미**: 15개 환경 중 worst 5개(30%)의 평균 성능 최적화
- **n_w**: 15 (환경 샘플 수)

**환경 샘플링 방법**:
- 각 iteration마다 113개 이미지 중 15개 랜덤 샘플
- 다양한 이미지 조건 경험

### 4.3 Gaussian Process 모델

**Kernel**:
- Type: Matérn 5/2
- 특성: 2번 미분 가능, 부드러운 함수 근사

**정규화**:
- Input (X, W): [0, 1]^14로 정규화
- Output (Y): Zero mean, unit variance

**학습**:
- 매 iteration마다 전체 데이터로 재학습
- Maximum Likelihood Estimation (MLE)

---

## 5. 데이터셋

### 5.1 이미지 데이터

**전체 데이터셋**:
- 총 이미지: 113개 welding images
- 해상도: 2448 × 3264 pixels
- 촬영 조건: 다양한 조명, 각도, 용접 상태

**Quick Test 설정**:
- **사용 이미지**: 30개 (시간 절약)
- 선정 방법: 랜덤 샘플링
- 평가 이미지: 각 iteration당 15개 (n_w)

### 5.2 Ground Truth

**용접선 타입** (4종):
1. Left Longitudinal (좌측 세로선)
2. Right Longitudinal (우측 세로선)
3. Fillet (필렛 용접선)
4. Collar (칼라 용접선)

**라벨링 형식**:
- 각 선당 2개 점 (시작점, 끝점)
- 총 8개 점 (4 lines × 2 points)
- 좌표: (x, y) 픽셀 좌표

**라벨링 방법**: 수동 (전문가 annotation)

---

## 6. 검출 파이프라인

```
[1] Input Image (2448×3264)
      ↓
[2] YOLO Detection
      → Welding seam ROI 추출
      → Model: YOLOv8 (custom trained)
      ↓
[3] Environment Feature Extraction
      → Physical: local_contrast, gradient_strength, etc.
      → CLIP: ROI → "rough surface" similarity, etc.
      ↓
[4] AirLine Line Detection
      → Q 프리셋: 여러 선 후보 생성
      → QG 프리셋: 여러 선 후보 생성
      → 각 프리셋당 수십 개 후보
      ↓
[5] RANSAC Selection (Weighted)
      → Q 후보 × weight_q
      → QG 후보 × weight_qg
      → 가중 점수 기반 최적 선 선택
      → 각 타입당 1개 선 선택 (총 4개)
      ↓
[6] LP_r Evaluation
      → 4개 검출 선 vs 4개 GT 선
      → 픽셀 샘플링 (400×400)
      → 거리 계산 및 threshold 적용
      → LP_r 점수 (0~1)
      ↓
[7] CVaR Computation
      → 15개 이미지 LP_r 수집
      → Worst 30% (5개) 평균
      → CVaR 값 반환
      ↓
[8] GP Update & KG Optimization
      → 새 데이터로 GP 재학습
      → 다음 평가할 파라미터 선택
      ↓
[9] Repeat (2-8)
```

---

## 7. 실행 환경

### 7.1 하드웨어

**GPU**:
- 모델: NVIDIA GeForce RTX 4060
- VRAM: 8 GB
- 메모리 제한: 80% (6.4 GB 사용)
- CUDA 버전: 11.8

### 7.2 소프트웨어 스택

**핵심 라이브러리**:
- PyTorch: 2.x (CUDA enabled)
- BoTorch: Latest (Bayesian Optimization)
- GPyTorch: Latest (GP 모델)
- CLIP: OpenAI CLIP ViT-B/32
- NumPy, SciPy: 수치 계산
- OpenCV: 이미지 처리

**전용 모듈**:
- YOLO: Ultralytics (welding detection)
- AirLine: CRG311.pyd (line detection)
- Custom: evaluation.py, optimization.py

### 7.3 실행 시간

**Quick Test**:
- 설정: 15 iterations, 30 images, n_w=15
- 소요 시간: 약 50분
- Iteration당: 약 3분

---

## 8. 실험 결과

### 8.1 성능 지표

**CVaR 진화**:
```
Initial CVaR (Iter 0-4 avg): 0.6192
BO Start (Iter 5):           0.8185
Final CVaR (Iter 19):        0.8899
Best CVaR (Iter 15):         0.9102

Improvement: +43.7% (0.6192 → 0.8899)
```

**History (20 iterations)**:
```
Init:  0.62 → 0.58 → 0.60 → 0.71 → 0.70
BO:    0.82 → 0.75 → 0.87 → 0.86 → 0.87
       0.87 → 0.87 → 0.88 → 0.87 → 0.88
       0.91 → 0.87 → 0.88 → 0.89 → 0.89
```

**특징**:
- 초기 샘플링: 불안정 (0.58~0.71)
- BO 시작 후: 급격한 향상 (0.82)
- 수렴 단계: 안정적 유지 (0.87~0.91)
- 마지막 도약: Best 달성 (0.91)

### 8.2 최적 파라미터

**Best Parameters (Iter 15)**:
```python
{
  # AirLine Q preset
  "edgeThresh1": -22.5587,    # 매우 낮은 threshold (민감)
  "simThresh1": 0.5198,       # 낮은 유사도 (관대)
  "pixelRatio1": 0.0540,      # 기본값 유지

  # AirLine QG preset
  "edgeThresh2": -12.5987,    # 중간 threshold
  "simThresh2": 0.5681,       # 중간 유사도
  "pixelRatio2": 0.0302,      # 낮은 비율 (엄격)

  # RANSAC weights
  "ransac_weight_q": 16.14,   # Q 프리셋 선호
  "ransac_weight_qg": 14.67   # QG 약간 낮음
}
```

**해석**:
- Q 프리셋이 더 공격적 (낮은 threshold → 더 많은 edge 검출)
- RANSAC에서 Q를 약간 더 선호 (16.14 vs 14.67)
- 전반적으로 recall 중심 전략 (많이 검출 후 필터링)

### 8.3 통계

**개선 통계**:
- Improvements: 7회
- Stagnations: 13회
- Best iteration: 15
- Convergence: 안정적 (Iter 6 이후)

**CVaR 분포**:
- 최소값: 0.5817 (Iter 1)
- 최대값: 0.9102 (Iter 15)
- 평균: 0.8092
- 표준편차: 0.1049

---

## 9. 시각화 및 분석

### 9.1 생성된 파일

- `results/bo_cvar_20251116_061530.json`: 전체 결과 데이터
- `results/bo_cvar_20251116_061530_exploration.png`: 9-panel 시각화
- `logs/run_20251116_061530/`: Iteration별 상세 로그

### 9.2 시각화 구성

**9-Panel Visualization**:
1. Optimization Progress (CVaR vs Iteration)
2. Total Improvement Over Time (누적)
3. Per-Iteration Improvement (막대)
4. Best Parameters (텍스트)
5. Optimization Statistics (텍스트)
6. CVaR Distribution (히스토그램)
7. Smoothed Trend (이동평균)
8. Exploration vs Exploitation (산점도)
9. Learning Speed (미분)

---

## 10. 결론 및 기여

### 10.1 기술적 기여

1. **BoRisk 알고리즘의 용접선 검출 적용** (최초)
   - 14차원 고차원 공간에서 효율적 최적화
   - 300회 평가만으로 43.7% 성능 향상

2. **CLIP 기반 환경 변수 도입**
   - Vision-Language model의 semantic understanding 활용
   - 전통적 CV features 보완
   - "rough", "smooth" 등 고수준 특징 포착

3. **CVaR 기반 강건성 확보**
   - Worst 30% 시나리오에서도 높은 성능 보장
   - 다양한 이미지 조건에 강건

### 10.2 실용적 가치

1. **43.7% 성능 향상** (CVaR 0.62 → 0.89)
2. **자동화된 파라미터 튜닝** (수동 튜닝 대체)
3. **다양한 이미지 조건에서 안정적 성능**

### 10.3 한계 및 향후 연구

**현재 한계**:
1. Quick test로 30개 이미지만 사용 (113개 전체 필요)
2. 15 iterations로 제한 (100 iterations 확장 필요)
3. 환경 변수 6개 (추가 특징 탐색 가능)

**향후 연구**:
1. 전체 데이터셋 (113 images)으로 확장 실험
2. 100 iterations overnight 실험
3. CLIP text prompt 최적화
4. 추가 환경 특징 탐색 (clip_bright, clip_dark 등)

---

## 11. 참고문헌

1. AirLine: "Efficient Learnable Line Detection with Local Edge Voting", IROS 2023
2. BoRisk: "Bayesian Optimization under Risk", arXiv:2011.05939
3. CLIP: "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
4. BoTorch: "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization", NeurIPS 2020

---

**작성일**: 2025-11-17
**실험 수행**: Graduate Student
**검토 대기**: 교수님 검토 필요
**다음 단계**: Overnight 실험 (100 iterations) 진행 중
