# Session 13 - 결론 및 다음 액션

**Date**: 2025-11-14
**Status**: 40+ iterations 진행 중, 조기 종료 권장
**Next**: Opus 제안 전략 구현

---

## 📊 실험 결과 요약

### 성능
- **Initial CVaR**: 0.4787
- **Best CVaR**: 0.5654 (Iter 9)
- **Current CVaR**: ~0.47 (40+ iterations)
- **개선율**: +18% (Iter 9 기준), 이후 30회 정체

### 문제점
1. **KG 예측 실패**: correlation = -0.253 (음수)
2. **Iter 9 이후 개선 없음**: 30회 정체
3. **Sobol 효과 미미**: 랜덤 샘플링보다 오히려 나쁨

---

## 🔍 원인 분석

### 가설 1: 외삽 문제 (Extrapolation) ❌

**검증 결과:**
```
BO 샘플 vs Initial 샘플:
  - 정규화 거리: 평균 20.7% (MODERATE, 크지 않음)
  - 먼 샘플 성능: 0.499 (더 좋음!)
  - 가까운 샘플 성능: 0.456 (오히려 나쁨!)

→ 외삽이 주요 문제가 아님!
```

### 가설 2: 환경 차원 과다 (Too Many Dims) ⚠️

**증거:**
- 14D 공간 (8D params + 6D env)
- 교차 항 고려 시: 8×6 = 48D 효과
- 200개 샘플로 부족할 수 있음

**하지만:**
- 200개면 14D에 충분 (rule of thumb: 140개)
- 문제는 차원이 아니라 **데이터 분포**

### 가설 3: CVaR 계산 문제 ✅ (주요 원인)

**핵심 문제:**
```python
# optimization.py에서 CVaR 계산
cvar = best_x_among_history의 GP_predicted_CVaR

# 하지만 로그에 기록되는 score는
score = current_iteration의 actual_evaluation

→ CVaR ≠ Score (완전히 다른 X!)
```

**증거:**
- Best CVaR (0.5654, Iter 9): 실제 Score 0.5127
- Best Score (0.8112, Iter 1): CVaR 0.4787
- Correlation (CVaR, Score) = -0.072 (거의 무관!)

**결론:**
- GP의 CVaR 예측이 부정확
- KG가 잘못된 목표 최적화
- 환경 효과(W) 예측 실패가 원인

---

## 💡 Opus 제안 전략

### 1. 환경 특징 축소 (6D → 4D)

**선택:** Top 4 features (|r| >= 0.35)
- local_contrast (r = -0.510)
- clip_rough (r = -0.454)
- brightness (r = -0.364)
- clip_smooth (r = +0.341)

**효과:**
- 14D → 12D (파라미터 8D + 환경 4D)
- 교차 항 48D → 32D (33% 감소)
- 강한 특징만 유지 → 노이즈 제거

### 2. Warm Start 전략

**Phase 1: Warm Start (환경 없음)**
```python
# n_initial = 20
# 파라미터만 최적화 (8D)
# 전체 이미지에서 CVaR 계산
# → 좋은 파라미터 영역 찾기
```

**Phase 2: BO with Environment**
```python
# iterations = 50
# 파라미터 + 환경 (12D)
# Phase 1에서 찾은 좋은 X 영역에서
# 환경 효과(W) fine-tuning
```

**이론적 근거:**
- Warm start로 좋은 X 영역을 먼저 탐색
- 좋은 X에서 W 관계 학습 → 외삽 거리 감소
- Multi-fidelity BO와 유사한 개념

### 3. Alpha 조절 실험

- alpha = 0.2 (worst 20%)
- alpha = 0.3 (worst 30%, 현재)
- alpha = 0.4 (worst 40%)

---

## 🎯 다음 액션

### 즉시 할 일 (Priority 1)

**1. 현재 실험 조기 종료**
```bash
# Session 13 실험 중단 (50회 정도에서)
pkill -f optimization.py
```

**2. Environment Top 4 파일 생성**
```bash
cd BO_optimization

python -c "
import json

with open('environment_top6.json') as f:
    data = json.load(f)

top4 = ['local_contrast', 'clip_rough', 'brightness', 'clip_smooth']

data_top4 = {
    img: {k: v for k, v in feat.items() if k in top4}
    for img, feat in data.items()
}

with open('environment_top4.json', 'w') as f:
    json.dump(data_top4, f, indent=2)

print(f'Created environment_top4.json with {len(data_top4)} images')
"
```

**3. Warm Start 구현**

필요한 수정:
- `optimization.py`에 `warm_start_initialization()` 추가
- `--warm_start` argument 추가
- GP 초기화 로직 수정

**4. 테스트 실행**
```bash
# 빠른 테스트 (5 initial + 3 iterations)
python optimization.py \
    --warm_start \
    --n_initial 5 \
    --iterations 3 \
    --env_file environment_top4.json \
    --alpha 0.3
```

**5. 본 실험**
```bash
# Warm start + Top 4 environment
python optimization.py \
    --warm_start \
    --n_initial 20 \
    --iterations 50 \
    --env_file environment_top4.json \
    --alpha 0.3 \
    --n_w 15
```

**예상 시간:**
- 구현: 1-2시간
- Warm start: 1-2시간 (20×113 평가)
- BO: 2-3시간 (50 iterations)
- Total: 4-7시간

---

### 차선책 (Priority 2)

만약 Warm start 구현이 복잡하면:

**Option A: 환경 제거 (8D만)**
```bash
python optimization.py \
    --no_environment \
    --iterations 50 \
    --alpha 0.3
```
- 가장 안전
- 이전 성공 케이스 (CVaR 0.6886) 재현

**Option B: Top 4만 사용 (현재 방식)**
```bash
python optimization.py \
    --iterations 50 \
    --env_file environment_top4.json \
    --n_initial 20  # 증가
    --alpha 0.3
```
- Warm start 없이도 개선 가능
- n_initial 증가로 초기 탐색 강화

---

## 📝 구현 가이드 (간단)

### environment_top4.json 생성

```bash
cd BO_optimization

python << 'EOF'
import json

with open('environment_top6.json') as f:
    data_top6 = json.load(f)

top4_features = ['local_contrast', 'clip_rough', 'brightness', 'clip_smooth']

data_top4 = {}
for img_name, features in data_top6.items():
    data_top4[img_name] = {
        k: v for k, v in features.items()
        if k in top4_features
    }

with open('environment_top4.json', 'w') as f:
    json.dump(data_top4, f, indent=2)

print(f"✓ Created environment_top4.json")
print(f"  Images: {len(data_top4)}")
print(f"  Features per image: {len(data_top4[list(data_top4.keys())[0]])}")
print(f"  Features: {list(data_top4[list(data_top4.keys())[0]].keys())}")
EOF
```

### Warm Start 구현 (핵심만)

```python
# optimization.py에 추가

def warm_start_phase(images_data, n_initial=20, alpha=0.3):
    """Phase 1: 환경 없이 파라미터만 최적화"""
    print("\n" + "="*70)
    print("PHASE 1: Warm Start (Parameters Only, No Environment)")
    print("="*70)

    # Sobol 샘플링 (8D params)
    sobol = SobolEngine(dimension=8, scramble=True)
    X_candidates = sobol.draw(n_initial)
    X_params = BOUNDS[0] + X_candidates * (BOUNDS[1] - BOUNDS[0])

    Y_warmstart = []

    for i, x in enumerate(X_params):
        print(f"\nWarm start {i+1}/{n_initial}")

        # 전체 이미지 평가
        scores = []
        for img_data in images_data:
            score = evaluate_single(x, img_data)
            scores.append(score)

        # CVaR 계산
        y = compute_cvar(torch.tensor(scores), alpha=alpha)
        Y_warmstart.append(y)
        print(f"  CVaR: {y:.4f}")

    Y_warmstart = torch.tensor(Y_warmstart).unsqueeze(-1)

    print(f"\n✓ Warm start complete")
    print(f"  Best CVaR: {Y_warmstart.max():.4f}")
    print(f"  Mean CVaR: {Y_warmstart.mean():.4f}")

    return X_params, Y_warmstart

# main()에서
if args.warm_start:
    X_ws, Y_ws = warm_start_phase(images_data, args.n_initial, args.alpha)
    # Phase 2: 환경 포함 BO (구현 필요)
    ...
else:
    # 기존 방식
    ...
```

---

## ⚠️ 주의사항

### 1. GP 차원 불일치 문제

Phase 1: 8D (params)
Phase 2: 12D (params + env)

**해결책:**
- Warm start 데이터를 12D로 확장 (환경은 평균값 또는 0으로 패딩)
- 또는 Phase 2에서 새 GP 생성

### 2. 시간 소요

- Warm start 20개: 약 20×113×5초 = 2시간
- 너무 길면 n_initial=10으로 축소

### 3. 검증 필요

- Warm start CVaR이 0.6+ 나오는지 확인
- 안 나오면 전략 재검토

---

## 📊 예상 결과

### 보수적 예상

```
Warm Start (Phase 1):
  Best CVaR: 0.62 (환경 없음, 8D 최적화)

BO with Env (Phase 2):
  Improvement: +5%
  Final CVaR: 0.65

→ Session 13 (0.565) 대비 +15% 개선
```

### 낙관적 예상

```
Warm Start:
  Best CVaR: 0.68 (충분한 초기 탐색)

BO with Env:
  Improvement: +10%
  Final CVaR: 0.75

→ 목표 달성!
```

---

## 🎓 배운 교훈

### 1. 외삽 문제가 아니었다

- 거리 분석 결과: 먼 샘플이 오히려 성능 좋음
- 문제는 외삽이 아니라 **CVaR 예측 실패**

### 2. 환경 상관의 역설

- 약한 상관 (r=0.12): GP가 무시 → 성공 (CVaR 0.69)
- 중간 상관 (r=0.33): GP가 학습 시도 → 실패 (CVaR 0.57)
- → 중간이 가장 나쁨!

### 3. Sobol의 한계

- Sobol은 샘플 커버리지 개선
- 하지만 GP 예측 정확도는 개선 못함
- 데이터 품질 > 샘플링 방법

### 4. 데이터 분포의 중요성

- 초기 X가 나쁘면 (성능 낮음)
- BO가 좋은 X 찾아도
- GP가 W 효과를 잘못 예측
- → Warm start로 해결 가능

---

## ✅ 체크리스트

**코딩 전:**
- [ ] Session 13 실험 중단
- [ ] environment_top4.json 생성 확인
- [ ] Warm start 로직 설계 검토
- [ ] NEXT_SESSION.md 업데이트

**구현:**
- [ ] `warm_start_phase()` 함수
- [ ] `--warm_start` argument
- [ ] GP 차원 전환 처리
- [ ] 테스트 실행 (n_initial=5, iter=3)

**실험:**
- [ ] 본 실험 시작 (n_initial=20, iter=50)
- [ ] Warm start CVaR 0.6+ 확인
- [ ] 최종 CVaR 0.65+ 확인
- [ ] 시각화 및 분석

**문서화:**
- [ ] 실험 결과 기록
- [ ] SESSION_14_SUMMARY.md 작성
- [ ] Git commit

---

**마지막 업데이트**: 2025-11-14 (40 iterations 분석 완료)
**상태**: 📋 Action Plan Ready
**다음**: environment_top4.json 생성 → Warm start 구현 → 실험
