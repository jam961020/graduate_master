# Overnight Experiment Settings
**Date**: 2025-11-17 19:15
**Status**: Retry with smaller settings

---

## 🔥 문제 발생

### 시도 1: 실패 (Exit 139 - Segmentation Fault)
```bash
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json
# 222개 이미지 on-the-fly 추출 중 크래시
```

### 시도 2: 실패 (Exit 127)
```bash
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 --max_images 113
# Initial sampling 중 크래시 (1/10 완료 후)
```

---

## ✅ 최종 설정 (시도 3)

### 안전한 설정으로 축소
```bash
python optimization.py \
  --iterations 50 \      # 100 → 50
  --n_initial 5 \        # 10 → 5
  --alpha 0.3 \
  --max_images 113 \
  --n_w 10               # 15 → 10
```

**변경 이유**:
- **iterations**: 100 → 50 (메모리 축적 방지)
- **n_initial**: 10 → 5 (초기 샘플링 부담 감소)
- **n_w**: 15 → 10 (환경 샘플링 수 감소)

**예상**:
- 초기 평가: 5 × 10 = 50
- BO 평가: 50 × 1 = 50
- 총 평가: 100
- 예상 시간: ~4-5시간

---

## 📊 내일 확인할 것

### 1. 실험 완료 확인
```bash
cd BO_optimization
ls -lt logs/ | head -3
ls logs/run_*/iter_*.json | wc -l
```

### 2. Best CVaR 확인
```python
import json, glob
files = sorted(glob.glob('logs/run_20251118_*/iter_*.json'))
cvars = [json.load(open(f))['cvar'] for f in files]
best_idx = cvars.index(max(cvars))
print(f'Best CVaR: {max(cvars):.4f} at Iter {best_idx+1}')
```

### 3. Visualization 생성
```bash
python visualization_exploration.py logs/run_20251118_XXXXXX
```

---

## 🎯 내일 할 일 우선순위

### Priority 1: 현재 결과 분석
- 50 iterations 결과 확인
- Quick test (3 iters) vs 현재 비교
- GP 안정성 검증

### Priority 2: 설정 최적화
만약 50 iters가 성공하면:
- [ ] 100 iterations로 재실험
- [ ] n_w = 15로 복원
- [ ] Full dataset 준비

### Priority 3: Environment Features 추출
- [ ] extract_environment_335.py 작성
- [ ] Batch 처리로 안전하게 추출
- [ ] 335장 실험 준비

---

## 💡 개선 아이디어

### 메모리 관리 강화
```python
# optimization.py에 추가
import gc

# 매 iteration 후
if iteration % 5 == 0:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Checkpoint 빈도 증가
```python
# 현재: 5 iterations마다
# 변경: 3 iterations마다 (안정성)
if (iteration + 1) % 3 == 0:
    save_checkpoint()
```

### Timeout 감지
```python
# 10분 동안 업데이트 없으면 경고
import time
last_update = time.time()

if time.time() - last_update > 600:
    print("WARNING: No progress for 10 minutes")
    save_checkpoint()
```

---

## 📝 트러블슈팅 로그

### 시도 1 (19:05)
- **명령**: 335장 full
- **결과**: Segmentation fault (Exit 139)
- **원인**: 222개 on-the-fly CLIP 추출

### 시도 2 (19:10)
- **명령**: 113장, 100 iters, 10 initial, 15 w
- **결과**: Exit 127
- **원인**: Initial sampling 중 크래시
- **진행**: Init 1/10 완료 후 멈춤

### 시도 3 (19:15) - 현재
- **명령**: 113장, 50 iters, 5 initial, 10 w
- **상태**: 실행 중...
- **로그**: 확인 대기

---

**마지막 업데이트**: 2025-11-17 19:15
**다음 확인**: 2025-11-18 09:00
