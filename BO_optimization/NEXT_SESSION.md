# Next Session Plan - Session 19
**Date**: 2025-11-18 (재부팅 후)
**Status**: 🛑 실험 대기 중 (재부팅 필요)

---

## 📋 현재 상황 요약

### SESSION 18에서 완료된 작업 ✅

#### 1. GP 붕괴 문제 진단 및 수정 (Critical Fix!)
**문제**:
- 이전 실험 (run_20251117_111151, 83 iterations)에서 GP 모델 붕괴
- Best CVaR: 0.7662 (Iter 61) → Final: 0.2672 (Iter 83)

**원인**:
1. Environment features (6D) 정규화 안 됨 → GP 수치 불안정
2. GP noise level 너무 낮음 (0.007) → Overfitting

**수정 사항**:
- `optimization.py` Line 313-329: 환경 벡터 정규화 추가
- `optimization.py` Line 770-773, 900-902, 917-919: GP noise constraint [0.001, 0.1]

**검증 결과**:
- Quick test (5 images, 3 iterations) 성공
- Init: 0.2967 → 0.3400
- Iter 1: 0.6953 ⭐
- Iter 3: 0.7410 ⭐ (+149.8% improvement!)
- Log: `logs/run_20251118_010827/`

✅ **GP 정상 작동 확인!**

---

#### 2. 데이터셋 확장
- `dataset/ground_truth_auto.json`: **335개 라벨** ✅
- `dataset/images/test/`: 336개 이미지 ✅
- `dataset/images/test2/`: **1031개 추가 이미지** 복사 완료 ✅
- 총 예상: ~1200장 (auto-labeling 후)

---

#### 3. Overnight 실험 시도 (모두 실패)

**시도 1**: 335장 전체
```bash
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json
```
- **결과**: Segmentation Fault (Exit 139)
- **원인**: 222개 이미지 on-the-fly CLIP 추출 중 GPU 메모리 초과

**시도 2**: 113장, 100 iterations
```bash
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 --max_images 113
```
- **결과**: Exit 127
- **원인**: Initial sampling 중 크래시 (1/10 완료 후)

**시도 3**: 113장, 축소 설정 (계획)
```bash
python optimization.py --iterations 50 --n_initial 5 --alpha 0.3 \
  --max_images 113 --n_w 10
```
- **상태**: ⚠️ 시작 안 함 (사용자가 재부팅 후 직접 실행 예정)

---

### 생성된 문서 ✅

1. **SESSION_18_PROGRESS.md**
   - 전체 작업 로그
   - GP 수정 사항 상세
   - 실험 실패 원인 분석
   - 코드 변경 내역

2. **NEXT_SESSION_PLAN.md** (이전 버전)
   - 다음 세션 계획 (상세)
   - Environment 추출 가이드
   - 트러블슈팅 절차

3. **OVERNIGHT_SETTINGS.md**
   - 3번의 실험 시도 로그
   - 실패 원인 및 설정 변경 이유
   - 최종 설정 (50 iters, 5 initial, 10 w)

4. **visualization_exploration.py** (새로 작성)
   - 9-panel 종합 분석 그래프
   - 이전 실험 (83 iters) 분석 완료

---

## 🎯 재부팅 후 즉시 할 일

### Step 1: 환경 확인
```bash
# Conda 환경 활성화 (재부팅 후 필수!)
conda activate weld2024_mk2

# 작업 디렉토리
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

# Python 및 PyTorch 확인
python --version
python -c "import torch; print(torch.__version__)"
```

---

### Step 2: 실험 시작 (2가지 옵션)

#### Option A: 안전한 설정으로 먼저 테스트 (추천!)
**113장, 축소 설정**:
```bash
python optimization.py \
  --iterations 50 \
  --n_initial 5 \
  --alpha 0.3 \
  --max_images 113 \
  --n_w 10
```

**예상**:
- 초기 평가: 5 × 10 = 50
- BO 평가: 50 × 1 = 50
- 총 평가: 100
- 시간: ~4-5시간
- 안정성: 높음 ✅

**완료 후**:
- 결과 확인
- 문제 없으면 100 iterations로 확장

---

#### Option B: Environment Features 먼저 추출 (안전 최우선!)
**335장 전체를 안전하게 사용하려면**:

1. **Environment 추출 스크립트 작성**:
```python
# extract_environment_335.py
#!/usr/bin/env python
"""335장 이미지의 environment features 추출 (배치 처리)"""

import torch
import json
from pathlib import Path
import gc
from environment import get_clip_model, extract_environment_features
from PIL import Image

# CLIP 모델 로드
clip_model, preprocess = get_clip_model()

# 이미지 목록
image_dir = Path('../dataset/images/test')
images = sorted(list(image_dir.glob('*.jpg')))

print(f"Total images: {len(images)}")

# Batch 처리 (10개씩)
batch_size = 10
all_features = {}

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    batch_num = i//batch_size + 1
    total_batches = (len(images) + batch_size - 1) // batch_size

    print(f'\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} images...')

    for img_path in batch:
        try:
            # PIL로 이미지 로드
            img = Image.open(img_path).convert('RGB')

            # Features 추출 (environment.py 함수 사용)
            features = extract_environment_features(str(img_path), clip_model, preprocess)

            # 저장 (image_name: features dict)
            all_features[img_path.stem] = features

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f'  Progress: {len(all_features)}/{len(images)} completed')

# JSON 저장
output_file = 'environment_335.json'
with open(output_file, 'w') as f:
    json.dump(all_features, f, indent=2)

print(f'\n✅ Saved: {output_file} ({len(all_features)} images)')
print(f'   Features per image: {list(list(all_features.values())[0].keys())}')
```

2. **실행**:
```bash
python extract_environment_335.py
```
- 시간: ~30분-1시간
- 안전: GPU 배치 처리로 안정적

3. **335장 실험 시작**:
```bash
python optimization.py \
  --iterations 100 \
  --n_initial 10 \
  --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json \
  --env_file environment_335.json
```
- 시간: ~10-12시간
- 안정성: 매우 높음 ✅✅

---

## 📊 결과 확인 방법

### 실험 진행 중 모니터링
```bash
# 실시간 모니터링
python monitor_progress.py logs/run_YYYYMMDD_HHMMSS

# 또는 수동 확인
watch -n 30 'ls -lt logs/run_*/iter_*.json | head -5'

# 최신 iteration 확인
ls -lt logs/run_*/iter_*.json | head -3

# CVaR 추이
tail -5 logs/run_*/iter_*.json | jq '.iteration, .cvar'
```

### 실험 완료 후
```bash
# 완료 확인
ls logs/run_*/iter_*.json | wc -l

# Best CVaR 찾기
python -c "
import json, glob
files = sorted(glob.glob('logs/run_*/iter_*.json'))
cvars = [(json.load(open(f))['iteration'], json.load(open(f))['cvar']) for f in files]
best = max(cvars, key=lambda x: x[1])
print(f'Best: Iter {best[0]}, CVaR={best[1]:.4f}')
"

# Visualization 생성
python visualization_exploration.py logs/run_YYYYMMDD_HHMMSS
```

---

## 🔧 다음 단계 (실험 완료 후)

### Priority 1: 결과 분석
- [ ] Best CVaR 및 parameters 확인
- [ ] Visualization 생성 (9-panel)
- [ ] 이전 실험들과 비교:
  - SESSION 15: 30 images, Best CVaR 0.9102
  - SESSION 17: 113 images, Best CVaR 0.7662 (GP 붕괴)
  - SESSION 18: Quick test, Best CVaR 0.7410
  - SESSION 19: 현재 실험

### Priority 2: Environment Features 335장 추출
**만약 Option A로 시작했다면**:
- [ ] `extract_environment_335.py` 작성 및 실행
- [ ] 335장 실험 시작

### Priority 3: Auto-labeling (test2 이미지)
- [ ] `auto_labeling.py` 작성
  - AirLine로 6개 점 자동 추출
  - ground_truth.json 포맷으로 저장
- [ ] 1031장 라벨링 완료
- [ ] 전체 ~1200장 데이터셋 준비

### Priority 4: Full-scale 실험
- [ ] 1200장 environment features 추출
- [ ] 최종 실험 (100-200 iterations)
- [ ] 논문 Results section 작성

---

## 🚨 트러블슈팅

### Segmentation Fault 재발 시
```bash
# Option 1: 이미지 수 더 줄이기
python optimization.py --max_images 50 --iterations 50

# Option 2: Environment features 미리 추출 (위 Option B)

# Option 3: GPU 메모리 제한 강화 (optimization.py 수정)
# Line 31-35
torch.cuda.set_per_process_memory_fraction(0.5)  # 80% → 50%
```

### Exit 127 또는 프로세스 크래시
```bash
# 설정 더 축소
python optimization.py \
  --iterations 30 \
  --n_initial 3 \
  --alpha 0.3 \
  --max_images 50 \
  --n_w 5
```

### Python 환경 문제
```bash
# Conda 환경 재활성화
conda deactivate
conda activate weld2024_mk2

# PyTorch 재설치 (필요시)
pip install torch torchvision --force-reinstall
```

---

## 📁 중요 파일 위치

```
BO_optimization/
├── optimization.py                    # ✅ GP 정규화 수정됨
├── visualization_exploration.py       # ✅ 9-panel viz
├── monitor_progress.py                # ✅ 실시간 모니터링
├── extract_environment_335.py         # 🔜 작성 필요 (Option B)
│
├── environment_top6.json              # 113 images
├── environment_335.json               # 🔜 생성 예정
│
├── logs/
│   ├── run_20251117_111151/          # SESSION 17 (83 iters, GP 붕괴)
│   ├── run_20251118_010827/          # SESSION 18 Quick test (성공!)
│   └── run_20251118_XXXXXX/          # 🔜 다음 실험
│
├── results/
│   ├── visualization_exploration_*.png
│   └── bo_cvar_*.json
│
├── SESSION_15_EXPERIMENT_REPORT.md   # 이전 성공 실험
├── SESSION_18_PROGRESS.md            # 오늘 진행사항
├── OVERNIGHT_SETTINGS.md             # 실험 시도 로그
└── NEXT_SESSION.md                   # 이 파일!

dataset/
├── ground_truth_auto.json             # 335 labels ✅
├── images/
│   ├── test/                         # 336 images ✅
│   └── test2/                        # 1031 images ✅ (라벨 없음)
```

---

## 💡 추천 작업 순서

### 🥇 가장 안전한 경로 (추천!)
1. **재부팅 후 환경 확인**
2. **Option A: 113장, 50 iterations** (~5시간)
3. **결과 확인 및 분석**
4. **문제 없으면 100 iterations로 재실험** (~10시간)
5. **Environment 335장 추출** (~1시간)
6. **335장 실험** (~12시간)

**총 소요**: 2-3일

---

### 🥈 시간 절약 경로
1. **재부팅 후 환경 확인**
2. **Option B: Environment 335장 추출** (~1시간)
3. **335장 실험 바로 시작** (~12시간)
4. **다음날 결과 분석**

**총 소요**: ~1일

---

### 🥉 최소 검증 경로
1. **Quick test 재실행** (이미 성공했으니 재확인)
```bash
python optimization.py --iterations 3 --n_initial 2 --alpha 0.3 --max_images 5 --n_w 3
```
2. **문제 없으면 바로 113장 50 iterations**
3. **이후 Option A 경로 따라가기**

---

## 🎯 최종 목표

### 이번 주 내
- ✅ GP 수정 완료 (정규화 + noise constraint)
- ✅ Quick test 검증 완료
- ✅ 335장 GT 준비 완료
- ✅ 1031장 추가 이미지 준비
- ⏳ **113장 또는 335장 안정적 실험 완료**
- ⏳ **Visualization 생성**

### 다음 주
- [ ] test2 auto-labeling (1031장)
- [ ] 1200장 full-scale 실험
- [ ] 논문 Results section 작성

---

## 📈 예상 성능

### Quick Test (SESSION 18)
- Images: 5
- Iterations: 3
- **Best CVaR: 0.7410** (+149.8%)
- ✅ GP 수정 검증 완료!

### 이전 실험 (SESSION 15)
- Images: 30
- Iterations: 20
- **Best CVaR: 0.9102** (+43.7%)
- 참고: GP 수정 전

### 예상 (다음 실험)
- Images: 113 or 335
- Iterations: 50-100
- **예상 Best CVaR: 0.75-0.85**
- GP 안정성 확보로 더 나은 수렴 기대

---

## 🔍 핵심 체크리스트

### 재부팅 후 반드시 확인
- [ ] `conda activate weld2024_mk2`
- [ ] `cd /c/Users/.../BO_optimization`
- [ ] `python --version` (3.12.0)
- [ ] `import torch` 정상 작동

### 실험 시작 전 확인
- [ ] GPU 메모리 충분 (RTX 4060 8GB)
- [ ] Dataset 경로 확인
- [ ] GT 파일 확인 (335 labels)
- [ ] Environment JSON 확인 (113 or 335)

### 실험 중 모니터링
- [ ] 10분마다 새 iteration 파일 생성 확인
- [ ] CVaR 값 정상 범위 (0.3-0.9)
- [ ] 프로세스 실행 중 확인
- [ ] GPU 메모리 overflow 없음

### 완료 후 분석
- [ ] 전체 iteration 개수 확인
- [ ] Best CVaR 및 parameters 추출
- [ ] Visualization 생성
- [ ] 이전 실험들과 비교

---

**작성일**: 2025-11-18 01:20
**상태**: 재부팅 대기 중
**다음 작업**: 환경 확인 → Option A 또는 B 선택 → 실험 시작

**🌙 굿나잇! 내일 아침 좋은 결과 기대합니다!**
**재부팅 후 이 파일부터 읽고 시작하세요!**
