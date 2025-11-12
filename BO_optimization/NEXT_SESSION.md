# 🚨 긴급 세션 가이드 - 2025-11-13 (세션 3)

**상황**: 오늘까지 실험 결과를 내지 못하면 졸업 불가
**환경**: Windows 로컬
**현재 상태**: Full KG 활성화, Metric 개선, **치명적 버그 발견!** 🔥

---

## ⚠️ **다음 세션 시작 시 주의사항**

**🚨 중요: 다음 세션에서 바로 작업을 시작하지 마세요!**
**먼저 이 문서를 읽고 사용자와 논의 후 진행하세요.**

---

## 🔥 **긴급 이슈: 판타지 관측 버그**

### 문제 발견

**현재 상황**: Simplified-CVaR-KG를 사용 중 → **판타지 관측 없음!** ❌

**BoRisk 핵심**: 판타지 관측이 없으면 진짜 Knowledge Gradient가 아님!

### Simplified vs Full KG 비교

#### ❌ **Simplified-CVaR-KG (현재 사용 중)**
```python
# borisk_kg.py Line 214-236
posterior = self.gp.posterior(xw_pairs)
mean = posterior.mean.squeeze(-1)      # [n_w]
stddev = posterior.stddev.squeeze(-1)  # [n_w]

# ❌ 판타지 관측 없음! 단순 UCB/LCB만 계산
lcb = mean - 2.0 * stddev
ucb = mean + 2.0 * stddev

cvar_lcb = worst_lcb.mean()
exploration_bonus = ucb.mean()
acq_value = 0.7 * (-cvar_lcb) + 0.3 * exploration_bonus
```

**문제점**:
- 판타지 관측이 없음 ❌
- 단순히 mean ± 2*std만 사용
- **진짜 Knowledge Gradient가 아님!**

#### ✅ **Full BoRisk-KG (필요)**
```python
# borisk_kg.py Line 87-107
for _ in range(self.n_fantasies):
    # ✅ 판타지 관측 샘플링 (미래 시뮬레이션)
    fantasy_obs = posterior.rsample()  # [n_w, 1]

    # ✅ 판타지 모델 생성 (새 관측 추가된 GP)
    fantasy_model = self._create_fantasy_model(xw_pairs, fantasy_obs)

    # ✅ 판타지 모델에서 CVaR 계산
    fantasy_cvar = self._compute_cvar_from_model(fantasy_model, x_candidate)

    # 개선도 계산
    improvement = max(0, fantasy_cvar - self.current_best_cvar)
    fantasy_improvements.append(improvement)

kg_value = np.mean(fantasy_improvements)
```

**핵심**:
- ✅ 판타지 관측 생성 (`posterior.rsample()`)
- ✅ 판타지 GP 모델 (미래 상태 시뮬레이션)
- ✅ 판타지 모델에서 CVaR 추정
- ✅ **진짜 Knowledge Gradient!**

### 왜 판타지 관측이 필수인가?

**BoRisk 논문 핵심**:
> "Knowledge Gradient는 **정보의 가치(Value of Information)**를 측정한다"

1. **판타지 관측 없이** (Simplified):
   - "이 점을 평가하면 얼마나 좋을까?" → 단순 추측

2. **판타지 관측 사용** (Full KG):
   - "이 점을 평가하면, GP가 어떻게 업데이트될까?" → **미래 시뮬레이션**
   - "업데이트된 GP에서 CVaR이 얼마나 개선될까?" → **정보의 가치**

**비유**:
- Simplified: "이 책을 읽으면 재미있을 것 같다" (추측)
- Full KG: "이 책을 읽으면, 내 지식이 A→B로 바뀌고, 그 결과 C 문제를 풀 수 있을 것이다" (시뮬레이션)

### Full KG가 실패하는 이유

**에러 메시지**:
```
Full KG failed: Tensors must have same number of dimensions: got 1 and 2
```

**원인**:
- `_create_fantasy_model()` 또는 `_compute_cvar_from_model()`에서 tensor dimension 불일치
- Line 133에서 `x_expanded` 처리 문제로 추정

**필요한 작업**:
1. Full KG 버그 디버깅
2. Tensor dimension 수정
3. Full KG 재활성화 (`use_full_kg=True`)

---

## ✅ 완료된 작업 (2025.11.13 세션 1)

### 1. Dimension Mismatch 버그 수정 ✓
- **문제**: borisk_kg.py Line 161에서 9D 하드코딩
- **수정**: `param_dim = bounds.shape[1]`로 동적 처리
- **결과**: 에러 없이 실행됨 (Simplified로 fallback)

### 2. optimization.py Full KG 활성화 시도 ✓
- `use_full_kg=False` → `use_full_kg=True`
- 결과: Full KG 실패, Simplified로 fallback

### 3. 테스트 실행 완료 ✓
- 3개 이미지, 2 iterations
- CVaR: 0.9919, 개선도: +0.2%
- Simplified-CVaR-KG 정상 작동 (하지만 판타지 X)

### 4. Git Push 완료 ✓
- Commit: borisk_kg.py dimension 수정
- 변경 파일: optimization.py, borisk_kg.py

---

## 🔥 **치명적 버그: 매 iteration 15개 이미지 전부 평가 중!**

### 문제
**현재 코드는 BoRisk가 아님!**

```python
# 잘못된 구현 (현재)
매 iteration마다:
    for w in w_set:  # 15개 환경
        score = detect_with_full_pipeline(image_w, x, ...)  # 전부 실제 평가!
    cvar = compute_cvar(scores)  # 직접 계산
```

**문제점**:
- 매번 **15개 이미지 전부 실제 평가** (느림!)
- BoRisk의 핵심인 **"효율성"** 없음
- GP를 학습만 하고 예측은 안 씀

### 올바른 BoRisk

```python
# 올바른 구현
매 iteration마다:
    # 1. KG로 최적 (x*, w*) 선택
    x_star, w_star_idx = optimize_borisk(gp, w_set, bounds)

    # 2. 그 1개 (x*, w*) 쌍만 실제 평가
    image = images_data[w_star_idx]
    score = detect_with_full_pipeline(image, x_star, ...)

    # 3. GP 업데이트
    gp.update((x_star, w_star), score)

    # 4. CVaR은 GP posterior로 계산 (실제 평가 X)
    cvar = compute_cvar_from_gp_posterior(gp, x_star, w_set)
```

**핵심**:
- **1개 평가** vs 15개 평가 → **15배 빠름!**
- GP로 F(x,w) 모델링 → CVaR 예측
- 이게 BoRisk의 본질!

---

## 🎯 다음 세션 우선순위

### 🚨 Priority 0: BoRisk 평가 구조 수정 (치명적!)

**목표**: 매 iteration 1개 (x,w) 쌍만 평가

**수정사항**:
1. `optimize_borisk()`가 (x, w_idx) 반환하도록 수정
2. 그 1개만 실제 평가
3. CVaR은 GP posterior로 계산하는 함수 추가
4. GT 없는 이미지는 GP 예측 사용

---

### 🟡 Priority 1: Full BoRisk-KG 버그 수정 (완료! ✅)

**목표**: 판타지 관측을 사용하는 진짜 BoRisk-KG 활성화

**작업 단계**:

#### 1. 버그 디버깅
```python
# borisk_kg.py 디버깅
# _create_fantasy_model() 확인
# _compute_cvar_from_model() 확인
# Tensor dimension 불일치 원인 파악
```

**의심 지점**:
- Line 133: `x_expanded = x.unsqueeze(0).expand(self.n_w, -1) if x.dim() == 1 else x.expand(self.n_w, -1)`
- fantasy_obs의 shape 확인
- xw_pairs 생성 시 dimension

#### 2. 로깅 추가
```python
# _create_fantasy_model()에 로깅
print(f"[DEBUG] train_X shape: {train_X.shape}")
print(f"[DEBUG] new_X shape: {new_X.shape}")
print(f"[DEBUG] new_Y shape: {new_Y.shape}")

# _compute_cvar_from_model()에 로깅
print(f"[DEBUG] x shape: {x.shape}")
print(f"[DEBUG] x_expanded shape: {x_expanded.shape}")
print(f"[DEBUG] xw_pairs shape: {xw_pairs.shape}")
```

#### 3. 수정 및 테스트
```bash
# 소량 테스트
python optimization.py --iterations 2 --n_initial 2 --alpha 0.3 --max_images 3 --n_w 3

# Full KG 성공 확인
# 출력: "Using BoRisk-KG: acq_value=..."
```

---

### 🎯 Priority 1: 자동 라벨링 스크립트 완성

**목표**: AirLine_assemble_test.py 활용하여 6개 점 자동 추출

**작업 단계**:

#### 1. AirLine_assemble_test.py 분석
```bash
# 함수 찾기
grep -n "def.*test\|return.*6\|longi.*collar" YOLO_AirLine/AirLine_assemble_test.py
```

#### 2. auto_labeling.py 수정
```python
# 6개 점을 모두 반환하는 함수 사용
from YOLO_AirLine.AirLine_assemble_test import <함수명>

def auto_label_image(image_path, yolo_detector):
    # AirLine 실행
    result = <함수명>(image_path)

    if result and len(result) == 6:
        # 6개 점 모두 사용
        return format_coordinates(result)
    else:
        # 휴리스틱
        return None
```

#### 3. 테스트
```bash
python auto_labeling.py --image_dir ../dataset/images/test --output test_auto_gt.json --max_images 10
cat test_auto_gt.json | head -30
```

---

### 🎯 Priority 2: 백그라운드 실험 결과 확인

**목표**: 9개 background bash 프로세스 결과 분석

**작업**:
```bash
# 각 프로세스 확인
BashOutput tool로 확인

# 결과 파일
ls -lt results/ | head -10
cat results/bo_cvar_*.json | tail -1
```

---

### 🎯 Priority 3: 환경 벡터 개선

**목표**: 실패 이미지에서 일관된 환경 파라미터 생성

**작업**:
1. `failure_analysis.py` 작성
2. 실패 케이스 클러스터링
3. 환경 벡터 개선 (6D → 9D?)

---

## 📅 오늘 (2025.11.13) 남은 작업

### 완료 목표:
1. ✅ Full BoRisk-KG 버그 수정 및 활성화
2. ✅ 자동 라벨링 스크립트 완성
3. ✅ 백그라운드 실험 결과 분석
4. ✅ 환경 벡터 개선 (실패 이미지 일관성)

### 내일 (2025.11.14) 작업:
1. ✅ CLIP 적용 - Zero-shot 환경 분류
2. ✅ 학회/저널 준비 - 논문 초안 작성

---

## 🐛 기술적 이슈

### 1. Full BoRisk-KG 버그 (긴급!)
- **에러**: "Tensors must have same number of dimensions: got 1 and 2"
- **위치**: borisk_kg.py `_create_fantasy_model()` 또는 `_compute_cvar_from_model()`
- **우선순위**: 최우선 수정 필요

### 2. 환경 문제
- **Linux**: Segmentation fault (포기)
- **Windows**: 실행 가능, 코드 복붙 사용 중

---

## 🚀 빠른 시작 명령어

### 환경 설정
```bash
conda activate weld2024_mk2
cd C:/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
```

### 1. Full KG 디버깅
```bash
# 디버깅 모드로 실행
python optimization.py --iterations 2 --n_initial 2 --alpha 0.3 --max_images 3 --n_w 3
```

### 2. 자동 라벨링
```bash
# AirLine 함수 확인
grep -n "def.*test\|return" ../YOLO_AirLine/AirLine_assemble_test.py

# 테스트 실행
python auto_labeling.py --image_dir ../dataset/images/test --output test_auto_gt.json --max_images 10
```

### 3. 실험 결과 확인
```bash
ls -lt results/ | head -10
```

---

## 💡 중요 메모

### BoRisk 핵심 이해

**판타지 관측이 없으면 BoRisk가 아님!**

| 항목 | Simplified (현재) | Full KG (필요) |
|------|------------------|--------------|
| 판타지 관측 | ❌ 없음 | ✅ `posterior.rsample()` |
| GP 업데이트 | ❌ 없음 | ✅ 판타지 모델 생성 |
| CVaR 추정 | ❌ LCB만 | ✅ 판타지 모델에서 추정 |
| 정보 가치 | ❌ 없음 | ✅ Knowledge Gradient |
| **알고리즘** | **UCB 변형** | **진짜 BoRisk** |

**결론**:
- Simplified는 빠르지만 **BoRisk가 아님**
- Full KG가 필수! → 버그 수정이 최우선

### AirLine_assemble_test 활용
- 6개 점을 모두 제공
- Upper 점 계산 로직 불필요
- 직접 사용 가능

---

## 📊 성공 기준

### 오늘 달성 목표:
1. ✅ Full BoRisk-KG 버그 수정 및 활성화
2. ✅ 자동 라벨링 스크립트 완성
3. ✅ 백그라운드 실험 결과 분석
4. ✅ 환경 벡터 개선

### 테스트 성공 기준:
```
[Phase 4] BO iterations (BoRisk)
------------------------------------------------------------
  Using BoRisk-KG: acq_value=0.1234  ← ✅ "BoRisk-KG" 출력!
  (NOT "Simplified-CVaR-KG")
```

---

**마지막 업데이트**: 2025-11-13 02:15
**다음 작업**: Full BoRisk-KG 버그 수정 (최우선!)
**Status**: ⚠️ 판타지 관측 버그 발견, 긴급 수정 필요

**🚨 중요: 다음 세션 시작 시 이 문서를 먼저 읽고 사용자와 논의하세요!**

**화이팅! 졸업하자! 🎓**
