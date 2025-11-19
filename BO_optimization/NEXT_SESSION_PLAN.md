# Next Session Plan - Session 19
**Date**: 2025-11-18 (Expected)
**Status**: 🌙 Overnight experiment running

---

## 🚀 현재 실행 중인 실험

### Experiment: 113 Images, 100 Iterations
```bash
Command: python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 --max_images 113
Start time: 2025-11-17 ~19:10
Expected completion: 2025-11-18 ~02:00 (약 7시간)
```

**설정**:
- Images: 113장 (environment_top6.json에 있는 것만)
- Iterations: 100
- Initial samples: 10
- CVaR alpha: 0.3
- w_set size: 15

**왜 113장?**:
- ✅ Environment features 이미 추출되어 있음
- ✅ Segmentation fault 없음
- ✅ 안정적으로 실행 가능
- ✅ Overnight 실험 적합

**예상 결과**:
- Log directory: `logs/run_20251118_XXXXXX/`
- Result file: `results/bo_cvar_20251118_XXXXXX.json`
- Best CVaR: ~0.75-0.85 (예상)

---

## 📋 다음 세션에서 할 일

### 1. 실험 결과 확인 ✅
```bash
cd BO_optimization

# 최신 log 디렉토리 확인
ls -lt logs/ | head -3

# 완료된 iteration 개수
ls logs/run_20251118_*/iter_*.json | wc -l

# Best CVaR 찾기
python -c "
import json, glob
files = sorted(glob.glob('logs/run_20251118_*/iter_*.json'))
cvars = [(json.load(open(f))['iteration'], json.load(open(f))['cvar']) for f in files]
best = max(cvars, key=lambda x: x[1])
print(f'Best: Iter {best[0]}, CVaR={best[1]:.4f}')
"

# Visualization 생성
python visualization_exploration.py logs/run_20251118_XXXXXX
```

### 2. Environment Features 추출 (335장) 🔧

**새 스크립트 작성 필요**: `extract_environment_335.py`

```python
#!/usr/bin/env python
"""
335장 이미지의 environment features 추출
Segmentation fault 방지를 위해 batch 처리
"""

import torch
import json
from pathlib import Path
import gc

# CLIP 모델 로드 (한 번만)
from environment import get_clip_model
clip_model, preprocess = get_clip_model()

# 이미지 목록 로드
image_dir = Path('../dataset/images/test')
images = sorted(list(image_dir.glob('*.jpg')))

# Batch 처리 (10개씩)
batch_size = 10
all_features = {}

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    print(f'Processing batch {i//batch_size + 1}/{len(images)//batch_size + 1}...')

    for img_path in batch:
        # Extract features
        features = extract_features(img_path, clip_model)
        all_features[img_path.stem] = features

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f'  Completed: {len(all_features)}/{len(images)}')

# Save to JSON
with open('environment_335.json', 'w') as f:
    json.dump(all_features, f, indent=2)

print(f'Saved: environment_335.json ({len(all_features)} images)')
```

**실행**:
```bash
cd BO_optimization
python extract_environment_335.py
```

### 3. 335장 실험 시작 🚀

**환경 features 추출 완료 후**:
```bash
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json \
  --env_file environment_335.json
```

**예상 시간**: ~10-12시간 (335장)

---

## 🔍 추가 분석 작업

### 결과 비교 테이블 작성

**비교 대상**:
1. SESSION 15: 30 images, Quick test (Best: 0.9102)
2. SESSION 17: 113 images, 83 iters (Best: 0.7662, GP 붕괴)
3. SESSION 18: 113 images, 100 iters (Overnight) ← **확인 필요**
4. SESSION 19: 335 images, 100 iters ← **예정**

**비교 항목**:
- Best CVaR
- Convergence speed (iterations to best)
- Final CVaR vs Best CVaR gap
- Improvement percentage
- Best parameters

### Visualization 개선

**추가 그래프**:
- Multiple runs comparison
- Parameter sensitivity analysis
- Environment feature importance
- w_set diversity analysis

---

## 📊 335장 실험 전 체크리스트

### ✅ 완료
- [x] GP 정규화 수정
- [x] GP noise constraint 추가
- [x] Quick test 검증 완료
- [x] 335장 GT 확인
- [x] 113장 overnight 실험 시작

### ⚠️ 진행 중
- [ ] 113장 실험 완료 대기

### 📝 대기
- [ ] Environment features 335장 추출
- [ ] 추출 스크립트 작성 및 실행
- [ ] 335장 실험 시작
- [ ] test2 이미지 (1031장) auto-labeling

---

## 🛠️ 코드 개선 사항 (선택)

### 1. Resume 안정성 향상
- Checkpoint 저장 주기 조정 (현재: 5 iterations)
- GPU 메모리 모니터링 추가
- Timeout 감지 및 자동 재시작

### 2. On-the-fly Extraction 개선
```python
# optimization.py Line 260-313 수정
# Batch 처리 추가
def extract_environment_batch(images, batch_size=10):
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        # Extract features
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 3. 모니터링 강화
- Real-time progress tracking
- Email/Slack notification
- Auto-restart on crash

---

## 📈 예상 타임라인

### 내일 아침 (09:00)
- [x] 113장 실험 결과 확인
- [x] Visualization 생성
- [x] Best parameters 분석

### 내일 오전 (10:00-12:00)
- [ ] Environment features 추출 스크립트 작성
- [ ] 335장 features 추출 실행 (~1-2시간)
- [ ] 추출 완료 확인

### 내일 오후 (13:00)
- [ ] 335장 실험 시작
- [ ] 모니터링 설정

### 모레 아침 (09:00)
- [ ] 335장 실험 결과 확인
- [ ] 전체 결과 비교 분석
- [ ] 논문 Figure 생성

---

## 🎯 최종 목표

### Short-term (이번 주)
1. ✅ 113장 안정적 실험 완료
2. 🔜 335장 실험 완료
3. 📊 결과 분석 및 visualization

### Mid-term (다음 주)
1. test2 이미지 auto-labeling (1031장)
2. Full dataset (1200장) environment features 추출
3. 1200장 실험 (최종)

### Long-term (논문)
1. 실험 결과 정리
2. Figure 생성 (9-panel visualization)
3. Table 작성 (비교 분석)
4. 논문 Results section 작성

---

## 💾 중요 파일 위치

```
BO_optimization/
├── optimization.py                    # ✅ 정규화 수정 완료
├── visualization_exploration.py       # ✅ 9-panel viz
├── monitor_progress.py                # ✅ Real-time monitoring
├── extract_environment_335.py         # 🔜 작성 필요
│
├── environment_top6.json              # 113 images
├── environment_335.json               # 🔜 생성 예정
│
├── logs/
│   ├── run_20251117_111151/          # SESSION 17 (83 iters, GP 붕괴)
│   ├── run_20251118_010827/          # Quick test (3 iters)
│   └── run_20251118_XXXXXX/          # 🔄 Overnight (100 iters)
│
├── results/
│   ├── visualization_exploration_*.png
│   └── bo_cvar_*.json
│
├── SESSION_15_EXPERIMENT_REPORT.md   # Quick test 보고서
├── SESSION_18_PROGRESS.md            # 오늘 진행사항
└── NEXT_SESSION_PLAN.md              # 이 파일
```

---

## 🚨 트러블슈팅 가이드

### 실험이 멈췄을 때
```bash
# 1. 최신 iteration 확인
ls -lt logs/run_20251118_*/iter_*.json | head -3

# 2. 마지막 업데이트 시간 확인 (10분 이상 지났으면 문제)
stat logs/run_20251118_*/iter_*.json | tail -10

# 3. Python 프로세스 확인
tasklist | grep python.exe

# 4. Resume
python optimization.py --resume_from logs/run_20251118_XXXXXX --iterations 100
```

### Segmentation Fault 발생 시
```bash
# Option 1: 이미지 수 줄이기
python optimization.py --max_images 50 --iterations 100

# Option 2: Environment features 미리 추출
python extract_environment_335.py
python optimization.py --env_file environment_335.json --iterations 100

# Option 3: Batch size 줄이기 (코드 수정 필요)
```

### GPU Out of Memory
```python
# optimization.py에서 메모리 제한 조정
# Line 31-35
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.set_per_process_memory_fraction(0.6)  # 80% → 60%
```

---

## 📞 빠른 참조

### 실험 시작
```bash
cd BO_optimization
conda activate weld2024_mk2

# 113장 (안전)
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 --max_images 113

# 335장 (features 추출 후)
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json \
  --env_file environment_335.json
```

### 결과 확인
```bash
# 최신 결과
ls -lt results/ | head -5

# Best CVaR
python -c "
import json, glob
files = sorted(glob.glob('logs/run_20251118_*/iter_*.json'))
best = max([json.load(open(f))['cvar'] for f in files])
print(f'Best CVaR: {best:.4f}')
"

# Visualization
python visualization_exploration.py logs/run_20251118_XXXXXX
```

### 모니터링
```bash
# Real-time
python monitor_progress.py logs/run_20251118_XXXXXX

# Manual
watch -n 30 'ls -lt logs/run_20251118_*/iter_*.json | head -5'
```

---

## 🎓 기술 노트

### GP 정규화 중요성
- **문제**: Environment features가 정규화되지 않으면 GP가 불안정
- **해결**: 모든 입력을 [0, 1]로 정규화
- **효과**: CVaR 안정화, 수렴 속도 향상

### CVaR vs Mean
- **Mean**: 평균 성능 (all scenarios)
- **CVaR (α=0.3)**: Worst 30% scenarios의 평균
- **장점**: Robustness, worst-case 성능 보장
- **단점**: 보수적, 최고 성능 희생 가능

### Knowledge Gradient
- **목적**: Information gain 최대화
- **특징**: Exploration + Exploitation balance
- **장점**: Sample efficiency, 빠른 수렴
- **단점**: 계산 비용 높음 (100 candidates × 15 w)

---

**작성일**: 2025-11-17 19:15
**실험 시작**: 2025-11-17 19:10 (113 images)
**예상 완료**: 2025-11-18 02:00
**다음 작업**: Environment 335 추출 → 335장 실험

**🌙 Good night! 내일 좋은 결과 기대됩니다!**
