# 일어났을 때 확인할 것들

**실험 시작 시각**: 2025-11-11 21:45 (한국시간)
**예상 완료 시간**: 약 2-4시간 후

---

## 1. 실험 진행 상태 확인

```bash
# 프로세스 상태 확인
ps aux | grep "python.*optimization.py" | grep -v grep

# 로그 실시간 확인
tail -f experiment_fullrun.log

# 로그 끝 50줄 확인
tail -50 experiment_fullrun.log
```

**예상 출력**:
- 실험 완료 시: `Best CVaR: X.XXXX` 같은 최종 결과
- 진행 중: `[Iteration XX/20]` 같은 중간 로그

---

## 2. 결과 파일 확인

```bash
# 로그 디렉토리 확인 (각 iteration 로그)
ls -lh logs/

# 결과 JSON 확인
ls -lt results/*.json | head -3

# 최신 결과 확인
cat results/bo_cvar_*.json | jq '.best_cvar' 2>/dev/null || cat results/bo_cvar_*.json
```

---

## 3. 성공 여부 판단

### ✅ 성공 조건
- [ ] `experiment_fullrun.log`에 "최적 파라미터:" 출력
- [ ] `logs/` 디렉토리에 `iter_001.json` ~ `iter_020.json` 파일 존재
- [ ] `results/` 디렉토리에 새 JSON 파일 생성
- [ ] CVaR 값이 0.0이 아님 (0.01 이상)

### ❌ 실패 시 확인
```bash
# 에러 메시지 확인
grep -i "error\|exception\|traceback" experiment_fullrun.log | head -20

# 마지막 에러
tail -100 experiment_fullrun.log | grep -A 10 -i "error"
```

---

## 4. 실험 완료 후 할 일

### 결과 분석
```bash
# 최종 CVaR 확인
grep "Best CVaR" experiment_fullrun.log

# iteration별 로그 확인
cat logs/iter_020.json | jq '.'

# 모든 iteration의 CVaR 추이
for f in logs/iter_*.json; do
  echo "$(basename $f): $(jq '.cvar' $f)"
done
```

### 시각화 (있다면)
```bash
ls results/*.png
```

---

## 5. 다음 단계

### 실험 성공 시
1. 결과 분석 및 정리
2. 더 많은 iteration으로 재실행 (30-50회)
3. 다른 alpha 값 테스트 (0.2, 0.4)
4. 결과를 논문에 반영

### 실험 실패 시
1. 로그에서 에러 원인 파악
2. 코드 수정 필요 여부 판단
3. 필요 시 Claude Code로 디버깅

---

## 6. 빠른 모니터링 스크립트

```bash
# 실험 진행률 체크
watch -n 60 'tail -20 experiment_fullrun.log'

# iteration 완료 개수 확인
watch -n 60 'ls logs/ | wc -l'
```

---

## 7. 문제 발생 시 긴급 조치

### 프로세스가 멈춘 경우
```bash
# PID 확인
ps aux | grep "python.*optimization.py" | grep -v grep

# 메모리 사용량 확인
free -h

# GPU 사용량 확인
nvidia-smi

# 필요 시 종료
kill <PID>
```

### 재실행
```bash
# 짧은 테스트
python -u optimization.py --iterations 5 --n_initial 5 --alpha 0.3

# 전체 재실행
nohup stdbuf -oL -eL python -u optimization.py --iterations 20 --n_initial 15 --alpha 0.3 > experiment_retry.log 2>&1 &
```

---

**마지막 업데이트**: 2025-11-11 21:45
**현재 상태**: 초기 샘플링 진행 중 (15개 샘플, 각 15개 환경 = 225 평가)
**PID**: 399948
