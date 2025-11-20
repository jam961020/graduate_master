# 다음 세션: 논문 작성 가이드

**작성일**: 2025-11-21
**실험 상태**: ✅ 완료
**다음 단계**: 📝 논문 작성

---

## 1. 완료된 실험 요약

### 실험 정보
- **Run ID**: `run_20251120_151025`
- **데이터셋**: 846장 (600 train / 246 validation)
- **실험 설정**: n_initial=10, n_iter=100, n_w=15, α=0.3, threshold=20px
- **소요 시간**: 약 3시간

### 핵심 결과
```
Initial Sampling Best:  CVaR = 0.5852
BO Start (Iter 1):      CVaR = 0.7020
Final Best (Iter 85):   CVaR = 0.7745 (+32.3% improvement)
Convergence (Iter 81-86): Mean 0.7735 ± 0.0005
```

### 생성된 자료
```
results/
├── convergence_plot_run_20251120_151025.png          # 수렴 그래프
├── visualization_exploration_run_20251120_151025.png # 9-panel 분석
└── visualization_bo_only_run_20251120_151025.png     # BO only

logs/run_20251120_151025/
└── iter_001.json ~ iter_100.json  # 전체 로그

BO_optimization/
├── PAPER_MATERIALS.md      # 논문 작성용 종합 자료
├── SESSION_26_SUMMARY.md   # 실험 과정 요약
├── validation_images.json  # 246장 검증 데이터 목록
└── environment_top6.json   # 환경 특징 (600장 중 113장)
```

---

## 2. 논문 작성을 위한 핵심 자료

### 2.1 Method 섹션용

#### 제안 기법 설명
- **문서**: `PAPER_MATERIALS.md` 섹션 2-3
- **내용**:
  - BoRisk 프레임워크 (CVaR 기반 최적화)
  - 8D 파라미터 공간 (AirLine 6D + RANSAC 2D)
  - 6D 환경 변수 (brightness, contrast, edge_density, etc.)
  - GP 모델 (14D 입력)
  - BoRisk-KG 획득 함수 (판타지 관측)
  - LP_r 평가 메트릭 (threshold=20px)

#### Algorithm 의사코드
- **위치**: `PAPER_MATERIALS.md` 섹션 3.1
- **내용**: Algorithm 1: BoRisk-based Parameter Optimization

#### 구현 세부사항
- **위치**: `PAPER_MATERIALS.md` 섹션 3.2-3.3
- **내용**: 데이터 분할, 환경 샘플링, GP 정규화, 코드 구조

### 2.2 Experiments 섹션용

#### 실험 설정
- **위치**: `PAPER_MATERIALS.md` 섹션 4
- **내용**:
  - 데이터셋: 846장 (600 train / 246 validation)
  - 하이퍼파라미터: n_initial=10, n_iter=100, n_w=15, α=0.3
  - 계산 환경: RTX 4060, Windows 10, Python 3.12
  - 실행 명령어

### 2.3 Results 섹션용

#### CVaR 수렴 결과
- **위치**: `PAPER_MATERIALS.md` 섹션 5.1-5.2
- **그래프**: `convergence_plot_run_20251120_151025.png`
- **핵심 수치**:
  ```
  Initial Sampling: [0.2915, 0.5852]
  BO Start: 0.7020
  Best: 0.7745 (Iter 85)
  Improvement: +32.3%
  Convergence: Iter 81-86
  ```

#### 최적 파라미터
- **위치**: `PAPER_MATERIALS.md` 섹션 5.3
- **파일**: `logs/run_20251120_151025/iter_085.json`
- **분석**: edgeThresh 낮음, simThresh2 높음, pixelRatio 높음, ransac_weight 균형

#### 실험 안정성
- **위치**: `PAPER_MATERIALS.md` 섹션 5.4
- **CVaR 변동성**: Initial (CV=22.6%) → Convergence (CV=0.1%)
- **Score=0 발생**: Session 25 (8회) → Session 26 (0회) ✓

### 2.4 시각화 자료

#### Figure 1: 시스템 개요
- **작성 필요**: Method 섹션용 다이어그램
- **내용**: Input Image → YOLO → AirLine → GP → BoRisk-KG → Next (x,w)

#### Figure 2: CVaR 수렴 그래프
- **파일**: `convergence_plot_run_20251120_151025.png`
- **사용처**: Results 섹션
- **설명**: Initial sampling (회색) + BO iterations (파란색) + Cumulative best (빨간 점선) + Convergence region (녹색)

#### Figure 3: 9-Panel 분석
- **파일**: `visualization_exploration_run_20251120_151025.png`
- **사용처**: Results 또는 Appendix
- **내용**: CVaR progress, cumulative best, improvement, distribution, etc.

#### Figure 4-6: 추가 필요 (선택사항)
- **Initial vs Final 비교**: 동일 이미지에서 검출 결과 비교
- **환경 특징 상관관계**: Pearson correlation heatmap
- **파라미터 진화**: 8D 파라미터의 iteration별 변화

---

## 3. 논문 구성 제안

### 3.1 Abstract (200-250 words)
```
[Background] 용접선 검출의 파라미터 튜닝 어려움, 환경 변화 취약
[Problem] 수동 튜닝의 한계, robust 성능 확보 어려움
[Method] BoRisk 프레임워크 + 환경 조건화 + GP 기반 최적화
[Results] 846장 이미지, 32.3% 성능 개선, 수렴 확인
[Contribution] 용접선 검출 최초 BoRisk 적용, robust parameter 자동 발견
```

### 3.2 Introduction (1.5-2 pages)
1. **배경**: 제조업 용접 품질 검사 중요성
2. **문제**: 기존 방법 한계 (수동 튜닝, 환경 민감성)
3. **목표**: Robust 파라미터 자동 최적화
4. **접근**: BoRisk + 환경 조건화 + GP
5. **기여**:
   - 용접선 검출 분야 최초 BoRisk 적용
   - 환경 조건화 통한 robust parameter 발견
   - 32.3% 성능 개선 달성

### 3.3 Related Work (1-1.5 pages)
1. **Welding Line Detection**: AirLine, 기존 알고리즘
2. **Bayesian Optimization**: GP, 획득 함수, BoTorch
3. **Robust Optimization**: CVaR, Risk-aware BO, BoRisk
4. **Environment-conditioned BO**: Contextual BO

### 3.4 Method (3-4 pages)
1. **Problem Formulation**: CVaR 최대화 문제 정의
2. **Parameter Space**: AirLine (6D) + RANSAC (2D) = 8D
3. **Environment Features**: 6D 시각적 특성 추출
4. **Gaussian Process Model**: 14D 입력, Matern 커널
5. **Acquisition Function**: BoRisk-KG, 판타지 관측
6. **Evaluation Metric**: LP_r, 직선 방정식 기반
7. **Algorithm**: 의사코드 (Algorithm 1)

### 3.5 Experiments (2-3 pages)
1. **Dataset**: 846장 (600 train / 246 validation)
2. **Implementation Details**:
   - 하이퍼파라미터
   - 계산 환경
   - 실행 명령어
3. **Evaluation Metrics**: CVaR, Score, Convergence

### 3.6 Results (2-3 pages)
1. **CVaR Improvement**: 0.5852 → 0.7745 (+32.3%)
   - Figure: 수렴 그래프
   - Table: Stage별 CVaR 비교
2. **Convergence Analysis**: Iter 81-86, Std=0.0005
   - Figure: 9-panel 분석
3. **Optimal Parameters**: Iteration 85 파라미터
   - Table: 최적 파라미터 값
   - 파라미터 분석 및 해석
4. **Stability Analysis**: CVaR 변동성 감소

### 3.7 Discussion (1-2 pages)
1. **Why it works**:
   - 환경 조건화 효과
   - BoRisk의 Risk-awareness
   - GP 기반 학습 효율성
2. **Strengths**:
   - 자동화
   - Robustness
   - 해석 가능성
3. **Limitations**:
   - 계산 비용
   - 환경 샘플링
   - 평가 메트릭
4. **Future Work**:
   - Validation 평가 (246장)
   - 환경별 성능 분석
   - 실제 배포

### 3.8 Conclusion (0.5 page)
- BoRisk 기반 용접선 검출 파라미터 최적화 성공
- 32.3% 성능 개선, 수렴 확인
- 자동화, Robustness 달성
- 다른 검출 알고리즘 확장 가능

---

## 4. 주요 수식

### CVaR 정의
```latex
\text{CVaR}_\alpha(X) = \mathbb{E}[X | X \leq \text{VaR}_\alpha(X)]
```

### BoRisk 목적 함수
```latex
x^* = \arg\max_x \text{CVaR}_\alpha[f(x, w)]
```

### Gaussian Process
```latex
f(x, w) \sim \mathcal{GP}(\mu(x, w), k((x, w), (x', w')))
```

### Knowledge Gradient
```latex
\text{KG}(x, w) = \mathbb{E}_y[\max_{x'} \mu_{n+1}(x', w') - \max_{x'} \mu_n(x', w')]
```

### LP_r 메트릭
```latex
\text{LP}_r(\text{detected}, \text{gt}) = \frac{1}{3} \sum_{i=1}^{3} \max\left(0, 1 - \frac{d_i}{\text{threshold}}\right)
```

---

## 5. Tables 계획

### Table 1: 파라미터 공간
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| edgeThresh1 | [-23, 7] | -3.0 | Q preset edge threshold |
| simThresh1 | [0.5, 0.99] | 0.98 | Q preset similarity |
| ... | ... | ... | ... |

### Table 2: 환경 특징
| Feature | Range | Calculation | Meaning |
|---------|-------|-------------|---------|
| brightness | [0, 1] | mean(gray) / 255 | Average brightness |
| contrast | [0, 1] | std(gray) / 128 | Contrast |
| ... | ... | ... | ... |

### Table 3: 실험 설정
| Parameter | Value | Description |
|-----------|-------|-------------|
| n_initial | 10 | Initial random sampling |
| n_iter | 100 | BO iterations |
| ... | ... | ... |

### Table 4: 성능 비교
| Stage | CVaR | Score | Improvement |
|-------|------|-------|-------------|
| Initial Best | 0.5852 | - | Baseline |
| BO Start | 0.7020 | - | +19.9% |
| Final Best | 0.7745 | 0.8941 | +32.3% |

### Table 5: 최적 파라미터 (Iter 85)
| Parameter | Value | Analysis |
|-----------|-------|----------|
| edgeThresh1 | -16.16 | Very low (weak edge detection) |
| simThresh1 | 0.76 | Medium (moderate merging) |
| ... | ... | ... |

### Table 6: CVaR 변동성
| Stage | Mean | Std Dev | CV (%) |
|-------|------|---------|--------|
| Initial (1-10) | 0.4343 | 0.0983 | 22.6% |
| Early BO (1-20) | 0.7094 | 0.0077 | 1.1% |
| ... | ... | ... | ... |

---

## 6. 참고 문헌

### 필수 참고문헌
1. **BoRisk**: Cakmak et al., "Bayesian Optimization of Risk Measures", NeurIPS 2020
2. **BoTorch**: Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization", NeurIPS 2020
3. **AirLine**: SAIR Lab, https://github.com/sair-lab/AirLine
4. **CVaR**: Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk", Journal of Risk 2000
5. **Knowledge Gradient**: Frazier et al., "The Knowledge-Gradient Policy for Correlated Normal Beliefs", INFORMS 2009
6. **Gaussian Process**: Rasmussen & Williams, "Gaussian Processes for Machine Learning", MIT Press 2006

### 추가 참고문헌
7. **PyTorch**: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019
8. **GPyTorch**: Gardner et al., "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration", NeurIPS 2018
9. **Contextual BO**: Kandasamy et al., "Multi-fidelity Bayesian Optimisation with Continuous Approximations", ICML 2017
10. **Robust BO**: Picheny et al., "Bayesian Optimization under Mixed Constraints", NeurIPS 2016

---

## 7. 논문 작성 체크리스트

### Phase 1: 초안 작성 (1-2일)
- [ ] Abstract 작성 (200-250 words)
- [ ] Introduction 작성 (1.5-2 pages)
- [ ] Related Work 작성 (1-1.5 pages)
- [ ] Method 작성 (3-4 pages)
  - [ ] Problem Formulation
  - [ ] Parameter Space
  - [ ] Environment Features
  - [ ] GP Model
  - [ ] Acquisition Function
  - [ ] Evaluation Metric
  - [ ] Algorithm 의사코드
- [ ] Experiments 작성 (2-3 pages)
  - [ ] Dataset
  - [ ] Implementation Details
  - [ ] Evaluation Metrics
- [ ] Results 작성 (2-3 pages)
  - [ ] CVaR Improvement
  - [ ] Convergence Analysis
  - [ ] Optimal Parameters
  - [ ] Stability Analysis
- [ ] Discussion 작성 (1-2 pages)
- [ ] Conclusion 작성 (0.5 page)

### Phase 2: Figure & Table 작성 (1일)
- [ ] Figure 1: 시스템 개요 (작성 필요)
- [x] Figure 2: CVaR 수렴 그래프 (완료)
- [x] Figure 3: 9-Panel 분석 (완료)
- [ ] Figure 4: Initial vs Final 비교 (선택)
- [ ] Figure 5: 환경 특징 상관관계 (선택)
- [ ] Figure 6: 파라미터 진화 (선택)
- [ ] Table 1-6: 모든 테이블 작성

### Phase 3: 수식 & 참고문헌 (0.5일)
- [ ] 모든 수식 LaTeX 형식으로 작성
- [ ] 참고문헌 BibTeX 정리
- [ ] Citation 확인

### Phase 4: 검토 & 수정 (1일)
- [ ] 문법 검토
- [ ] 논리 흐름 확인
- [ ] Figure & Table 번호 일치 확인
- [ ] 참고문헌 형식 통일
- [ ] 교수님 피드백 반영

---

## 8. 추가 분석 (선택사항)

### 8.1 Validation 평가 (선택)
**목적**: Overfitting 확인
**데이터**: `validation_images.json` (246장)
**방법**:
```bash
# 최적 파라미터로 246장 평가
python validate_best_params.py \
  --params_file logs/run_20251120_151025/iter_085.json \
  --validation_file validation_images.json
```

### 8.2 환경별 성능 분석 (선택)
**목적**: 다양한 환경에서 robustness 증명
**방법**:
- Brightness별 (Low/Medium/High): CVaR 비교
- Edge density별 (Low/High): CVaR 비교
- Blur level별 (Low/High): CVaR 비교

### 8.3 Baseline 비교 (선택)
**목적**: 개선 효과 강조
**비교 대상**:
- Default AirLine params vs Optimized params
- Random Search vs BoRisk
- Standard BO (EI) vs BoRisk

---

## 9. 실험 데이터 위치

### 로그 파일
```
logs/run_20251120_151025/
├── iter_001.json ~ iter_100.json  # 전체 iteration 로그
└── checkpoint_iter_080.json (if any)
```

### 결과 파일
```
results/
├── convergence_plot_run_20251120_151025.png
├── visualization_exploration_run_20251120_151025.png
└── visualization_bo_only_run_20251120_151025.png
```

### 데이터 파일
```
BO_optimization/
├── validation_images.json         # 246장 검증 데이터 목록
├── environment_top6.json          # 환경 특징 (113장)
└── ../dataset/
    ├── images/for_BO/             # 846장 이미지
    └── ground_truth_merged.json   # GT 라벨
```

### 문서 파일
```
BO_optimization/
├── PAPER_MATERIALS.md      # 📝 논문 작성용 종합 자료 (이거 보세요!)
├── SESSION_26_SUMMARY.md   # 실험 과정 요약
├── NEXT_SESSION_PAPER.md   # 이 파일
└── CLAUDE.md              # 개발 가이드
```

---

## 10. 논문 작성 시작 방법

### Step 1: 자료 확인
```bash
# 문서 읽기
code PAPER_MATERIALS.md  # 가장 중요!
code SESSION_26_SUMMARY.md

# 그래프 확인
open results/convergence_plot_run_20251120_151025.png
open results/visualization_exploration_run_20251120_151025.png

# 최적 파라미터 확인
cat logs/run_20251120_151025/iter_085.json
```

### Step 2: 논문 템플릿 작성
```latex
\documentclass{article}
\usepackage{neurips_2024}  % 또는 해당 학회 템플릿
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{algorithm,algorithmic}

\title{Risk-Aware Parameter Optimization for Robust Welding Line Detection}
\author{Your Name}

\begin{document}
\maketitle

\begin{abstract}
% PAPER_MATERIALS.md 섹션 8.1 참고
\end{abstract}

\section{Introduction}
% PAPER_MATERIALS.md 섹션 8.2 참고

% ... (계속)
\end{document}
```

### Step 3: 섹션별 작성
1. Abstract → Introduction → Related Work (배경 설명)
2. Method (핵심 기여)
3. Experiments → Results (실험 증명)
4. Discussion → Conclusion (해석 및 정리)

### Step 4: Figure & Table 삽입
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{convergence_plot.png}
\caption{CVaR convergence during optimization.
Initial sampling (gray) shows exploration,
while BO iterations (blue) demonstrate steady improvement
to the optimal CVaR of 0.7745 at iteration 85.}
\label{fig:convergence}
\end{figure}
```

---

## 11. 중요 메모

### ✅ 완료된 것
- BoRisk 기반 파라미터 최적화 (100 iterations)
- CVaR 32.3% 개선 달성 (0.5852 → 0.7745)
- 수렴 확인 (Iter 81-86, Std=0.0005)
- 시각화 자료 생성 (수렴 그래프, 9-panel 분석)
- 종합 문서 작성 (PAPER_MATERIALS.md)

### ❓ 선택사항 (하면 더 좋음)
- Validation 평가 (246장)
- 환경별 성능 분석
- Baseline 비교
- Initial vs Final 비교 Figure
- 환경 특징 상관관계 Figure

### 🎯 논문 작성 목표
- **목표 페이지**: 8-10 pages (ICRA/IROS 기준)
- **핵심 메시지**: "BoRisk로 robust한 용접선 검출 파라미터 자동 발견"
- **강점**: 자동화, Robustness, 32.3% 개선
- **마감**: [교수님께 확인]

---

**다음 세션에서 할 일**:
1. PAPER_MATERIALS.md 읽고 숙지
2. 논문 템플릿 작성 (LaTeX or Word)
3. 섹션별로 초안 작성
4. Figure & Table 정리
5. 교수님께 초안 검토 요청

**화이팅! 좋은 논문 쓰세요! 🎓📝**
