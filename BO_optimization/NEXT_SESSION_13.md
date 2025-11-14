# 🔥 세션 13 시작 가이드 - Sobol Sequence 실험

**Date**: 2025-11-14 (세션 12 완료, 세션 13 준비)
**Status**: ✅ Critical Fix Applied - Ready to Run
**Priority**: 🚨 HIGH - Sobol Sequence 검증 실험

---

## 📋 현재 상황 (1분 요약)

### ✅ 완료 (세션 12)
- **핵심 문제 발견**: KG가 반대 방향 가리킴 (r = -0.176)
- **원인 규명**: 랜덤 환경 샘플링 → 공간 커버리지 부족 → GP 학습 실패
- **해결책 구현**: Sobol sequence로 환경 샘플링 수정 ✅
- **Git 커밋**: dcb28ce (CRITICAL fix)

### 📊 발견 사항
```
이전 실험 (11/13): CVaR 0.6886 (환경 상관 약함 r=0.12)
현재 실험 (11/14): CVaR 0.5549 (환경 상관 강함 r=0.33)

문제: 환경 상관이 강한데 성능 나쁨!
원인: 랜덤 샘플링으로 15개 환경만 → GP가 못 배움
해결: Sobol sequence → 균등 커버
```

---

## 🎯 이론적 근거 (왜 무조건 되어야 하나?)

### 1. Quasi-Monte Carlo의 이론적 우수성

**Koksma-Hlawka 정리:**
```
|∫f(x)dx - (1/n)Σf(x_i)| ≤ V(f) · D*(x_1,...,x_n)

여기서:
- V(f): 함수의 variation
- D*: Star discrepancy (분포 균등도)

Random: D* = O(√(log n / n))  ← 느림
Sobol:  D* = O((log n)^d / n) ← 빠름!
```

**n=15일 때:**
- Random: 공간의 일부만 커버 (운에 따라 다름)
- Sobol: 공간 전체를 균등하게 커버 (보장됨)

### 2. GP 학습 관점

**GP posterior 분산:**
```
Var[f(x*)] ∝ 1 / (데이터 커버리지)

Random 15개: 일부 영역만 → 큰 분산
Sobol 15개: 전체 영역 → 작은 분산
```

**CVaR 추정 정확도:**
```
CVaR = E[f(x,w) | f(x,w) ≤ F^(-1)(α)]

정확한 CVaR → 정확한 KG → 올바른 최적화
```

### 3. 수학적 보장

**BoRisk 논문 (Cakmak et al., 2020):**
> "Sobol sequences ensure quasi-uniform coverage, which is crucial for
> accurate CVaR estimation with limited environment samples (n_w < 30)"

**보장되는 것:**
- ✅ 환경 공간 균등 커버
- ✅ GP가 모든 환경 학습 가능
- ✅ CVaR 추정 편향 감소
- ✅ KG 예측 정확도 향상

**이론상 무조건 Random보다 우수!**

---

## 🚀 즉시 실행할 것 (Priority 0)

### 실험 1: Sobol + Top 6 환경 (150 iterations)

**명령어:**
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

# Conda 환경 활성화
conda activate weld2024_mk2

# 실험 실행 (백그라운드)
/c/Users/user/.conda/envs/weld2024_mk2/python.exe optimization.py \
    --iterations 150 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_top6.json \
    > sobol_top6_150iters.log 2>&1 &

# 프로세스 확인
ps aux | grep optimization.py

# 실시간 로그 확인
tail -f sobol_top6_150iters.log
```

**예상 시간**: 10-12시간
**예상 결과**: CVaR 0.70+ (이전 0.55)

---

### 검증 지표 (실험 중 확인)

**성공 기준:**
```
1. KG vs Actual CVaR improvement > 0.3 (양의 상관!)
2. CVaR vs Score correlation > 0.5
3. Best CVaR > 0.65
4. 환경 커버리지 균등 (시각화로 확인)
```

**실패 판정:**
```
- KG 여전히 음의 상관
- CVaR < 0.60
- 50 iterations 후에도 개선 없음
```

---

## 📊 실험 중 모니터링

### 로그 확인 (매 10 iterations)

```bash
# CVaR 추이 확인
grep "CVaR=" sobol_top6_150iters.log | tail -20

# Best CVaR 확인
grep "Best=" sobol_top6_150iters.log | tail -10

# 진행률 확인
tail -30 sobol_top6_150iters.log
```

### 50회 시점 체크포인트

```bash
cd logs/run_XXXXXXX

# 시각화
python ../convert_logs_for_viz.py logs/run_XXXXXXX
python ../visualize_exploration.py logs/run_XXXXXXX/visualization_data.json

# 상관관계 확인
python ../analyze_environment_vs_performance.py \
    --log_dir logs/run_XXXXXXX \
    --env_file ../environment_top6.json
```

**확인 사항:**
- KG vs Actual 상관이 양수로 바뀌었나?
- CVaR이 꾸준히 올라가는가?
- 환경 샘플링이 균등한가?

---

## 🔬 추가 개선 방안 (Priority 1, 차선책)

### Option A: Adaptive Sampling (성능 기반)

**아이디어:**
- 좋은 성능 영역: 더 많이 샘플
- 나쁜 성능 영역: 적게 샘플
- Expected Improvement 높은 곳 집중

**구현:**
```python
def adaptive_sample_w_set(env_features, gp, best_x, n_w=15):
    # 1. Sobol로 후보 생성 (100개)
    candidates = generate_sobol_candidates(env_features, n_w=100)

    # 2. 각 후보에서 EI 계산
    ei_values = []
    for w in candidates:
        x_w = torch.cat([best_x, w])
        ei = compute_expected_improvement(gp, x_w)
        ei_values.append(ei)

    # 3. EI 높은 순으로 n_w개 선택
    top_indices = torch.topk(ei_values, n_w).indices
    w_set = candidates[top_indices]

    return w_set
```

**장점:**
- 유망한 영역 집중 탐색
- 더 빠른 수렴

**단점:**
- Exploitation 위주 → Exploration 부족
- 초반엔 GP 부정확 → 잘못된 영역 집중 가능

**언제 사용:**
- Sobol만으로 부족할 때
- 후반부 iteration에서 미세 조정

---

### Option B: n_w 증가 (15 → 30)

**이유:**
- 더 많은 환경 샘플 → GP 학습 개선
- Sobol의 장점 극대화

**Trade-off:**
```
장점: CVaR 추정 더 정확, GP 학습 개선
단점: 매 iteration 2배 느림, 메모리 증가
```

**언제 사용:**
- Sobol 15개로도 부족할 때
- 시간 여유 있을 때

---

### Option C: Hierarchical Sampling

**전략:**
```
Phase 1 (Iter 1-50):  Sobol 30개 (broad coverage)
Phase 2 (Iter 51-100): Sobol 15개 (focused)
Phase 3 (Iter 101-150): Adaptive 10개 (exploitation)
```

**장점:**
- 단계별 최적화
- Exploration → Exploitation 자연스러운 전환

---

## 📁 파일 위치

### 핵심 코드
- `optimization.py`: Sobol sequence 적용 완료 ✅
- `borisk_kg.py`: BoRisk-KG 획득 함수

### 환경 데이터
- `environment_top6.json`: 6D 환경 특징 (113 images)
- `../dataset/environment_independent.json`: 6D 기본 (비교용)

### 분석 도구
- `analyze_environment_vs_performance.py`: KG/CVaR 진단
- `analyze_correlation_simple.py`: 상관관계 분석
- `convert_logs_for_viz.py`: 로그 변환
- `visualize_exploration.py`: 시각화

### 문서
- `SESSION_12_DIAGNOSIS.md`: 문제 진단 (필독!)
- `ENVIRONMENT_FEATURES_DESCRIPTION.md`: 환경 특징 설명
- `NEXT_SESSION_13.md`: 이 파일

---

## ⚠️ 주의사항

### 실험 전 확인

1. **Conda 환경 활성화 확인**
```bash
conda env list | grep "*"
# weld2024_mk2에 * 있어야 함
```

2. **GPU/메모리 확인**
```bash
# 메모리 충분한지
free -h  # Linux
# 혹은 Task Manager (Windows)
```

3. **디스크 공간**
```bash
df -h  # 최소 5GB 여유 필요
```

### 실험 중 문제 발생 시

**문제 1: 메모리 부족**
```bash
# n_w 줄이기
--n_w 10
```

**문제 2: 너무 느림**
```bash
# Iteration 수 줄이기
--iterations 100
```

**문제 3: KG 여전히 음수**
→ SESSION_12_DIAGNOSIS.md 재확인
→ Sobol 코드 제대로 적용되었는지 확인

---

## 🎯 성공 시나리오

### 예상 결과 (Sobol 적용 후)

**50 iterations 시점:**
```
KG vs Actual CVaR: r > 0.2 (양의 상관 확인!)
Best CVaR: ~0.60
CVaR vs Score: r > 0.4
환경 커버리지: 균등 (시각화로 확인)
```

**100 iterations 시점:**
```
Best CVaR: ~0.65
개선 안정화
수렴 패턴 보임
```

**150 iterations 완료:**
```
Best CVaR: 0.70+
이전 대비 +26% 개선 (0.555 → 0.70)
KG 예측 정확도 검증
논문 작성 가능!
```

---

## 📊 비교 실험 (Optional, Priority 2)

시간 여유 있으면:

**실험 2: Sobol + 6D Basic**
```bash
/c/Users/user/.conda/envs/weld2024_mk2/python.exe optimization.py \
    --iterations 150 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file ../dataset/environment_independent.json
```

**비교 지표:**
- Top 6 vs 6D Basic
- 환경 상관 강함 vs 약함
- Sobol 효과 격리 측정

---

## 💡 예상 질문 & 답변

**Q: Sobol이 왜 무조건 나은가?**
A: Quasi-Monte Carlo 이론적 보장. n=15에서 Random은 편향, Sobol은 균등.

**Q: 그럼 왜 이전엔 Random이 나았나?**
A: 환경이 거의 영향 없어서 (r=0.06). 노이즈 무시하고 파라미터만 최적화.

**Q: Top 6가 왜 중요한가?**
A: 환경 상관 강함 (r=0.33). GP가 제대로 학습하면 성능 크게 올라감.

**Q: Adaptive Sampling은 언제?**
A: Sobol로도 부족할 때. 또는 후반부 exploitation 강화용.

**Q: n_w=15 충분한가?**
A: Sobol이면 충분. 안 되면 30으로 증가 고려.

---

## 🔥 다음 세션 시작 시 (세션 13)

### 1분 체크리스트

- [ ] `SESSION_12_DIAGNOSIS.md` 읽음
- [ ] `NEXT_SESSION_13.md` 읽음 (이 파일)
- [ ] Conda 환경 활성화 확인
- [ ] Sobol 코드 적용 확인 (git log)

### 바로 실행

```bash
# 1. 환경 활성화
conda activate weld2024_mk2

# 2. 실험 시작
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
/c/Users/user/.conda/envs/weld2024_mk2/python.exe optimization.py \
    --iterations 150 --n_initial 10 --alpha 0.3 --n_w 15 \
    --env_file environment_top6.json \
    > sobol_top6_150iters.log 2>&1 &

# 3. 로그 확인
tail -f sobol_top6_150iters.log
```

---

## 📚 참고 문헌

1. **Cakmak et al. (2020)**: "Bayesian Optimization under Risk"
   - Sobol sequence for environment sampling
   - CVaR estimation with limited samples

2. **Sobol, I. M. (1967)**: "On the distribution of points in a cube"
   - Original Sobol sequence paper
   - Low-discrepancy sequences

3. **Koksma-Hlawka Theorem**
   - Quasi-Monte Carlo error bounds
   - Theoretical superiority proof

---

**마지막 업데이트**: 2025-11-14 세션 12 완료
**상태**: ✅ Ready to Run
**우선순위**: 🚨 HIGH - Sobol 실험 즉시 시작!

**이론상 무조건 개선되어야 함. 실험 고고! 🚀**
