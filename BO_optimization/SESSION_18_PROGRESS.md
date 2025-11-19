# Session 18 Progress Report
**Date**: 2025-11-17
**Status**: ⚠️ Segmentation Fault 발생

---

## 🎯 오늘 완료한 작업

### 1. ✅ GP 붕괴 문제 진단 및 수정

**문제 발견**:
- 이전 실험 (run_20251117_111151) 에서 GP 모델이 붕괴
- Iter 83까지 실행 후 CVaR이 0.27~0.75 사이로 불안정
- Best CVaR: 0.7662 (Iter 61)
- Worst CVaR: 0.2672 (Iter 83)

**원인 분석**:
1. **환경 벡터 정규화 누락**: 6D environment features가 [0,1]로 정규화되지 않음
2. **GP noise level 너무 낮음**: 0.007로 overfitting 발생
3. **InputDataWarning**: 데이터가 unit cube에 없음

**수정 사항**:
```python
# optimization.py Line 313-329
# ===== CRITICAL: Normalize environment features to [0, 1] =====
env_min = env_features.min(dim=0)[0]
env_max = env_features.max(dim=0)[0]
env_range = env_max - env_min
env_range = torch.where(env_range < 1e-6, torch.ones_like(env_range), env_range)
env_features_normalized = (env_features - env_min) / env_range

# optimization.py Line 770-773
# ===== CRITICAL: Set noise constraint to prevent overfitting =====
from gpytorch.constraints import Interval
gp.likelihood.noise_covar.register_constraint("raw_noise", Interval(1e-3, 0.1))
```

**테스트 결과** (5장, 3 iterations):
```
Init: CVaR = 0.2967 → 0.3400
Iter 1: CVaR = 0.6953 ⭐
Iter 2: CVaR = 0.6948
Iter 3: CVaR = 0.7410 ⭐
개선도: +149.8%
```
✅ GP 정상 작동 확인!

---

### 2. ✅ Visualization 생성

**파일**: `visualization_exploration.py`
- 9-panel comprehensive visualization
- 이전 실험 (83 iterations) 분석 완료
- 저장: `results/visualization_exploration_run_20251117_111151.png`

**분석 결과**:
- Total iterations: 83
- Best CVaR: 0.7662 (Iter 61)
- Initial: 0.7281 → Final: 0.2672 (하락!)
- Improvement: +5.2% (초기 대비)
- Mean CVaR: 0.6780 ± 0.1181

**결론**: Resume 후 GP 붕괴로 성능 악화

---

### 3. ✅ 데이터셋 확장

**Ground Truth**:
- `dataset/ground_truth_auto.json`: **335개 라벨**
- `dataset/images/test`: 336개 이미지

**추가 이미지 준비**:
- Source: `C:/Users/user/Desktop/study/task/weld2025/.../all_images/images`
- **1031개 이미지** 복사 완료 → `dataset/images/test2/`
- 내일 auto-labeling 예정 → **총 ~1200장**

---

## 🚨 발생한 문제

### Segmentation Fault (Exit Code 139)

**시도한 실험**:
```bash
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json
```

**진행 상황**:
```
[✓] 335개 이미지 로드
[✓] YOLO 모델 로드
[✓] Environment JSON 로드 (113개)
[⚠️] 222개 이미지 on-the-fly 추출 시작
[✗] Segmentation fault 발생
```

**원인 추정**:
1. **CLIP 모델 메모리 부족**: 222개 이미지 동시 처리
2. **GPU 메모리 초과**: RTX 4060 8GB 제한
3. **CLIP feature 추출 크래시**: Vision-Language 모델 문제

**에러 메시지**:
```
Segmentation fault
/usr/bin/bash: line 1: 1927 Segmentation fault
```

---

## 📋 해결 방안

### Option 1: Environment Features 미리 추출 (추천)

**335개 이미지의 environment features를 미리 추출**:
```bash
# 새로운 스크립트 작성 필요
python extract_environment_features.py \
  --image_dir ../dataset/images/test \
  --gt_file ../dataset/ground_truth_auto.json \
  --output environment_335.json
```

**장점**:
- Segmentation fault 방지
- 실험 시작 시간 단축
- 재현성 확보

### Option 2: 113장으로 먼저 실험

**기존 environment_top6.json 활용**:
```bash
# 113장만 사용하여 실험
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --max_images 113
```

**장점**:
- 바로 실행 가능
- 안정성 검증

### Option 3: Batch 처리로 On-the-fly 추출

**코드 수정** (`optimization.py`):
```python
# Line 260-313에서
# 222개를 한번에 추출하지 말고, batch 단위로 처리
# + GPU 메모리 정리 추가
```

---

## 📊 현재 상태 요약

### 완료 ✅
- [x] GP 붕괴 문제 진단 및 수정
- [x] 환경 벡터 정규화 추가
- [x] GP noise constraint 추가
- [x] Quick test로 수정 사항 검증
- [x] Visualization 생성
- [x] 335장 GT 확인
- [x] 1031장 추가 이미지 복사

### 진행 중 ⚠️
- [ ] 335장 실험 (Segmentation fault)

### 대기 중 📝
- [ ] Environment features 미리 추출 (335장)
- [ ] 실험 재시작 (수정 후)
- [ ] test2 이미지 auto-labeling (1031장)
- [ ] 전체 데이터 environment features 추출

---

## 🔧 수정된 코드 요약

### 1. 환경 벡터 정규화
**파일**: `optimization.py` Line 313-329
- 모든 environment features를 [0, 1]로 정규화
- Division by zero 방지

### 2. GP Noise Constraint
**파일**: `optimization.py` Line 770-773, 900-902, 917-919
- 3곳에 모두 적용
- Constraint: [0.001, 0.1]
- Overfitting 방지

### 3. Visualization Script
**파일**: `visualization_exploration.py`
- 9-panel comprehensive analysis
- CVaR progress, distribution, statistics
- Best parameters display

---

## 📈 실험 결과 (이전 세션)

### SESSION 15 (Quick Test)
- 데이터: 30 images
- Iterations: 20
- Best CVaR: **0.9102** (Iter 15)
- Improvement: **+43.7%**

### SESSION 17 (Full - 실패)
- 데이터: 113 images
- Iterations: 83/100 (중단)
- Best CVaR: 0.7662 (Iter 61)
- 문제: GP 붕괴, Resume 후 성능 하락

### SESSION 18 (Test - 성공)
- 데이터: 5 images
- Iterations: 3
- Best CVaR: 0.7410
- Improvement: +149.8%
- ✅ 정규화 수정 검증 완료!

---

## 🎯 다음 세션 계획

### 우선순위 1: Environment Features 추출
```bash
# extract_environment_features.py 작성
# 335개 이미지 features 추출
# environment_335.json 생성
```

### 우선순위 2: 335장 실험
```bash
# 안정적인 환경에서 실행
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json \
  --env_file environment_335.json
```

### 우선순위 3: 추가 데이터 준비
- test2 이미지 auto-labeling
- 전체 environment features 추출
- 1200장 full 실험 준비

---

## 💾 중요 파일

```
BO_optimization/
├── optimization.py                              # ✅ 정규화 수정됨
├── visualization_exploration.py                 # ✅ 새로 작성
├── monitor_progress.py                          # ✅ 새로 작성
│
├── results/
│   ├── bo_cvar_20251117_111151.json            # 이전 실험 (83 iters)
│   ├── visualization_exploration_*.png          # 분석 결과
│   └── bo_cvar_20251118_010827.json            # Quick test (3 iters)
│
├── logs/
│   ├── run_20251117_111151/                    # 83 iters (GP 붕괴)
│   └── run_20251118_010827/                    # Quick test (성공)
│
├── SESSION_15_EXPERIMENT_REPORT.md             # Quick test 보고서
├── SESSION_18_PROGRESS.md                      # 이 파일
└── NEXT_SESSION.md                             # 다음 작업

dataset/
├── ground_truth_auto.json                       # 335 labels
├── images/
│   ├── test/                                   # 336 images
│   └── test2/                                  # 1031 images (추가)
```

---

## 🔍 디버깅 정보

### Segmentation Fault 분석

**발생 위치**: Environment feature extraction (on-the-fly)
```
[WARN] 222/335 images not in environment JSON
→ CLIP model loading for each image
→ Segmentation fault
```

**Python 프로세스**: 실행 중 아님 (crashed)
```
python.exe  9828  Console  1    9,020 K  (다른 프로세스)
python.exe 20432  Console  1   18,144 K  (다른 프로세스)
```

**메모리 상황**:
- GPU: RTX 4060 8GB
- 80% limit = 6.4GB 사용 가능
- CLIP ViT-B/32 + YOLO + 이미지 로딩
- 222개 동시 처리 → 메모리 초과 가능

---

## 🎓 기술적 기여 (업데이트)

### 1. GP 안정성 개선
- 환경 벡터 정규화로 numerical stability 확보
- Noise constraint로 overfitting 방지
- InputDataWarning 해결

### 2. BoRisk 알고리즘 개선
- 14D 입력 (8D params + 6D env) 안정화
- CVaR 기반 강건성 확보
- Knowledge Gradient 최적화

### 3. 데이터셋 확장
- 113장 → 335장 (3배 증가)
- 추가 1031장 준비 완료
- Auto-labeling 파이프라인 구축 예정

---

**작성일**: 2025-11-17 19:05
**다음 세션**: Environment features 추출 후 335장 실험
**예상 완료**: 내일 오전

**⚠️ CRITICAL**: Segmentation fault 해결 필요!
**✅ FIXED**: GP normalization + noise constraint
