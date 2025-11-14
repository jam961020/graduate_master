# 다음 세션 빠른 시작 가이드

## 🎯 현재 상황 (1분 요약)

**완료**: 6D BoRisk 50회 실행, Resume 기능 구현
**문제**: 150회 목표였으나 50회에서 중단 (원인 불명)
**결과**: CVaR 0.5114 → 0.5549 (+8.5% 개선)

---

## 🚀 바로 실행 (추천 순서)

### Option 1: Resume으로 이어서 (50→150회)

```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

# Conda 환경 활성화
conda activate weld2024_mk2

# 50회부터 이어서 100회 더 실행
/c/Users/user/.conda/envs/weld2024_mk2/python.exe optimization.py \
    --iterations 100 \
    --resume_from logs/run_20251114_044828 \
    --env_file environment_top6.json \
    --alpha 0.3 \
    --n_w 15 \
    > experiment_6d_resume.log 2>&1 &
```

**주의**: 현재 checkpoint가 14D라 resume 실패할 수 있음!
→ 실패하면 Option 2로

---

### Option 2: 새로 시작 (0→150회, 메모리 절약)

```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

conda activate weld2024_mk2

# n_w 줄여서 메모리 절약 (15→10)
/c/Users/user/.conda/envs/weld2024_mk2/python.exe optimization.py \
    --iterations 150 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 10 \
    --env_file environment_top6.json \
    > experiment_6d_v2.log 2>&1 &
```

**예상 시간**: 10-12시간

---

## 📊 결과 확인

### 실행 중 확인
```bash
# 진행 상황 보기
tail -f experiment_6d_resume.log

# CVaR 추이만 보기
grep "CVaR=" experiment_6d_resume.log | tail -20

# 프로세스 확인
ps aux | grep optimization.py
```

### 완료 후 분석
```bash
# 시각화
python visualize_results.py --log_dir logs/run_XXXXXXX

# 최적 파라미터 확인
grep "최적 파라미터" experiment_6d_resume.log -A 10

# CVaR 개선율
python -c "
import json
from pathlib import Path

log_dir = Path('logs/run_XXXXXXX')
files = sorted(log_dir.glob('iter_*.json'))

with open(files[0]) as f: first = json.load(f)
with open(files[-1]) as f: last = json.load(f)

print(f'First CVaR: {first[\"cvar\"]:.4f}')
print(f'Last CVaR: {last[\"cvar\"]:.4f}')
print(f'Improvement: {(last[\"cvar\"]-first[\"cvar\"])/first[\"cvar\"]*100:.1f}%')
"
```

---

## 🔍 중단 원인 디버깅 (Option 3)

먼저 왜 50회에서 멈췄는지 확인하고 싶다면:

```bash
# 1. 조기 종료 로직 확인
grep -n "조기 종료" optimization.py

# 2. 메모리 사용량 확인 (실행 중)
watch -n 5 'ps aux | grep python | grep optimization'

# 3. 로그 버퍼링 문제 확인
# → optimization.py에 flush=True 추가
```

---

## 📁 필수 파일 확인

```bash
cd BO_optimization

# 환경 파일
ls -lh environment_top6.json  # 6D 환경 (필수)

# 체크포인트
ls logs/run_20251114_044828/checkpoint_*.json  # 10개

# 결과
ls logs/run_20251114_044828/iter_*.json  # 50개
```

---

## 💡 팁

1. **n_w 조절**
   - 현재: 15 (기본)
   - 메모리 부족 시: 10 (33% 절약)
   - 빠른 테스트: 5

2. **Checkpoint 확인**
   - 매 5회마다 저장됨
   - Resume 실패 시 checkpoint 차원 확인

3. **로그 모니터링**
   - `tail -f` 로 실시간 확인
   - Iter XX/150 진행률 체크

---

## ❓ 자주 묻는 질문

**Q: Resume이 안 되는데요?**
A: Checkpoint가 옛날 버전(14D)일 수 있음 → Option 2로 새로 시작

**Q: 또 중단되면요?**
A:
1. n_w를 10으로 줄이기
2. Checkpoint 빈도 증가 (매 1회)
3. 로그에 flush=True 추가

**Q: 얼마나 기다려야 하나요?**
A:
- 100회: 6-8시간
- 150회: 10-12시간
- 매 iteration: 4-5분

---

**다음 세션 시작 전 읽을 파일**:
1. `QUICK_START_NEXT.md` (이 파일) ⭐️
2. `EXPERIMENT_STATUS.md`
3. `SESSION_11_SUMMARY.md`

**화이팅! 150회 완주하자! 🔥**
