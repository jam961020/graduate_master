# LP Metric 분석 및 원본 구현 (2025-11-16)

**작성일**: 2025-11-16
**상태**: AirLine 논문 원본 LP_r metric 구현 완료

---

## 📊 발단: Overnight 실험 (run_20251115_054348) 분석

### 실험 설정
- **Metric**: lp (F1 score, threshold=20px)
- **목표**: 100 iterations
- **실제**: 53 iterations (중단됨)

### 결과 요약

| 항목 | 값 | 상태 |
|------|-----|------|
| CVaR-Score correlation | 0.0747 | ❌ 거의 0 |
| Perfect score 비율 | 50.9% (27/53) | ❌ 너무 높음 |
| CVaR 평균 | 0.8969 | ✓ 안정적 |
| Session 13 대비 CVaR | +91.5% | ⚠️ 의심스러운 상승 |

### 주요 문제점

1. **실험 중단**: Iter 53에서 멈춤 (47개 부족)
   - InputDataWarning 108회 발생
   - BoTorch unit cube scaling 문제

2. **CVaR-Score correlation 여전히 0**
   - Session 13: r = -0.19 (음의 상관)
   - Overnight: r = 0.07 (거의 0)
   - Metric 변경해도 개선 없음

3. **Perfect score 과다**
   - threshold=50px → 20px로 줄였는데도 50.9% perfect
   - 변별력 여전히 부족

---

## 🔍 LP Metric 코드 분석

### 기존 구현 (evaluation.py)

```python
# Line 80-100
distances = cdist(gt_pixels, detected_pixels)
min_distances = distances.min(axis=1)
tp_count = np.sum(min_distances <= threshold)

# 🚨 버그 발견!
precision = tp_count / len(detected_pixels)  # ❌ 잘못된 계산
recall = tp_count / len(gt_pixels)           # ✅ 올바름

f1 = 2 * (precision * recall) / (precision + recall)
return f1  # ❌ F1 반환
```

### 버그 상세

**문제**: `precision` 계산이 잘못됨

- `tp_count`는 GT 픽셀 중 매칭되는 개수 (Recall의 분자)
- 이것을 `len(detected_pixels)`로 나누는 것은 precision이 아님!

**올바른 precision**:
```python
# 검출 픽셀 → GT 픽셀 방향으로 거리 계산
dist_det_to_gt = cdist(detected_pixels, gt_pixels)
min_dist_det = dist_det_to_gt.min(axis=1)
tp_precision = np.sum(min_dist_det <= threshold)
precision = tp_precision / len(detected_pixels)
```

**버그의 영향**:
```python
현재 precision = tp_count / len(detected_pixels)
              = (len(gt_pixels) * recall) / len(detected_pixels)
```

→ GT/Detected 픽셀 비율에 따라 precision이 왜곡됨

---

## 📚 AirLine 논문 확인

### 논문 정보
- **제목**: "AirLine: Efficient Learnable Line Detection with Local Edge Voting"
- **학회**: IROS 2023
- **arXiv**: https://arxiv.org/abs/2303.16500
- **GitHub**: https://github.com/sair-lab/AirLine

### LP_r (Line Precision) 정의

```
LP_r = Σ(τ_r(X) ⊗ Y) / ΣY
```

**여기서**:
- **X**: 검출된 선 (predicted lines)
- **Y**: Ground truth 픽셀 (GT line pixels)
- **τ_r**: dilation function with tolerance radius r
- **⊗**: element-wise multiplication (AND operation)

**의미**:
1. 검출된 선 X를 r 픽셀만큼 dilate
2. GT 픽셀 Y와 overlap 계산
3. **Overlap된 GT 픽셀 개수 / 전체 GT 픽셀 개수**

**→ LP_r은 사실상 Recall입니다!**

### 중요한 발견

1. **"Line Precision"이라는 이름이지만 실제로는 Recall**
   - GT coverage를 측정
   - Precision (검출 정확도)이 아님

2. **F1 score를 사용하지 않음**
   - AirLine 논문은 LP_r만 사용
   - Precision, F1 없음

3. **여러 tolerance 레벨 사용**
   - LP₀, LP₁, LP₂, LP₃, LP₅, LP₁₀
   - 다양한 tolerance로 robustness 평가

### 왜 Recall만 사용하는가?

**논문의 주장**:

1. **Endpoint 기반 metric의 한계**
   - 선의 길이, 방향 무시
   - 짧은 정확한 선들(LSD 등)을 낮게 평가

2. **주관적 라벨링 문제**
   - 수작업 라벨링의 일관성 부족
   - 비슷한 선에 다른 annotation

3. **Edge-to-line 일관성 중시**
   - 실제 로봇 응용에서 중요
   - GT를 얼마나 커버하는지가 핵심

---

## 💡 RANSAC 후 단일 선 → Over-detection 문제 최소화

### 사용자 지적 (매우 중요!)

**"RANSAC하면 단일 선만 남는데, 그 두 선의 픽셀을 비교하는 거잖아?"**

**정확합니다!**

### 우리 시스템의 특성

```
AirLine 검출
→ 여러 선 후보들 (Q, QG 프리셋)
→ RANSAC으로 대표 선 1개 선택
→ 최종 출력: 각 타입당 1개 선
   - Left Longi: 1개
   - Right Longi: 1개
   - Fillet: 1개
   - Collar: 1개
```

### Over-detection 문제가 없는 이유

**일반적인 line detection**:
```
검출: 수십~수백 개 선
GT: 4~10개 선
→ LP_r (Recall)만 사용 시 문제:
  - 엄청 많은 선 검출 → GT 전부 커버 → LP_r = 1.0
  - 하지만 False Positive 과다 → 실용성 없음
```

**우리 시스템 (RANSAC 후)**:
```
검출: 정확히 4개 선 (또는 3개, 검출 실패 시)
GT: 정확히 4개 선 (또는 3개)
→ 1:1 대응!
→ Over-detection 불가능
→ LP_r (Recall)만으로 충분!
```

### 결론

**LP_r (Recall only)이 우리 시스템에 적합한 이유**:

1. **RANSAC이 암묵적 Precision 보장**
   - 단일 선만 선택 → False Positive 최소화
   - 가중치 기반 선택 → 품질 보장

2. **1:1 대응 구조**
   - GT 선과 검출 선이 같은 개수
   - Precision/Recall 구별 불필요

3. **직관적 의미**
   - LP_r = GT 픽셀이 얼마나 잘 커버되었는가
   - 높을수록 정확한 검출

---

## ✅ 수정된 구현

### 새로운 evaluate_lp 함수

```python
def evaluate_lp(detected_coords, image, image_name=None, threshold=50.0, debug=False):
    """
    AirLine 논문의 LP_r (Line Precision) 구현

    LP_r = Σ(τ_r(X) ⊗ Y) / ΣY

    Returns:
        LP_r score (0~1): GT coverage ratio
    """
    # ... (픽셀 샘플링) ...

    # LP_r 계산: GT 픽셀 → 검출된 픽셀까지의 최소 거리
    distances = cdist(gt_pixels, detected_pixels)
    min_distances = distances.min(axis=1)

    # threshold r 이내에 있는 GT 픽셀 개수
    covered_gt_pixels = np.sum(min_distances <= threshold)

    # LP_r = covered GT pixels / total GT pixels
    lp_r = covered_gt_pixels / len(gt_pixels)

    return lp_r  # ✅ LP_r만 반환 (F1 아님!)
```

### 주요 변경점

1. **F1 score 제거** → LP_r (Recall)만 반환
2. **Precision 계산 제거** → 논문에 없음
3. **명확한 변수명** → `covered_gt_pixels`, `lp_r`
4. **상세한 docstring** → 논문 공식 명시
5. **RANSAC 특성 언급** → over-detection 문제 최소화

---

## 🎯 기대 효과

### 1. 올바른 Metric 사용

- **이전**: 잘못 구현된 F1 score
- **현재**: 논문 원본 LP_r (Recall)
- **효과**: 이론적으로 올바른 평가

### 2. RANSAC과의 시너지

- **RANSAC**: 단일 선 선택 (암묵적 Precision)
- **LP_r**: GT coverage 측정 (Recall)
- **결합**: 완전한 평가 시스템

### 3. 직관성 향상

```
LP_r = 0.95 → GT의 95%가 검출된 선으로부터 threshold 이내
           → 매우 정확한 검출

LP_r = 0.50 → GT의 50%만 커버
           → 검출 실패 또는 큰 오차
```

### 4. 비선형성 개선?

**이전 F1**:
- 잘못된 precision으로 인한 왜곡
- 비선형성 부족

**현재 LP_r**:
- 순수 Recall
- threshold 의존적
- 여전히 비선형성 부족할 수 있음

**→ threshold 조정으로 개선 가능**

---

## 📋 다음 단계

### Immediate (즉시)

1. **빠른 테스트** (10 iterations)
   ```bash
   python optimization.py \
       --iterations 10 \
       --n_initial 5 \
       --alpha 0.3 \
       --max_images 20
   ```

2. **확인 사항**:
   - [ ] CVaR-Score correlation 개선?
   - [ ] Score 분포 (0~1)
   - [ ] Perfect score 비율 감소?

### Short-term (단기)

3. **Threshold 실험**
   - threshold=50 (현재, 관대)
   - threshold=20 (중간)
   - threshold=10 (엄격)
   - threshold=5 (매우 엄격)

4. **비교 분석**
   - Session 13 vs Overnight vs 새 LP_r
   - CVaR progression
   - Correlation 변화

### Medium-term (중기)

5. **환경 제거 실험** (backup plan)
   ```bash
   python optimization.py \
       --no_environment \
       --iterations 50 \
       --alpha 0.3
   ```

6. **최종 실험** (LP_r + 최적 설정)

---

## 🔬 배운 것

### 1. 논문 원본 확인의 중요성

- 구현이 논문과 다를 수 있음
- "Line Precision"이라는 이름에 속지 말 것
- 실제 공식을 확인해야 함

### 2. Metric 이름의 함정

- **LP (Line Precision)** → 실제로는 Recall!
- **F1 score** → 논문에서 사용 안 함
- 이름보다 정의가 중요

### 3. RANSAC의 역할 재인식

- 단순히 노이즈 제거가 아님
- **암묵적 Precision 보장**
- Recall metric과 완벽한 조합

### 4. Over-detection vs Under-detection

- **일반 line detection**: Over-detection 문제 심각
- **우리 시스템**: RANSAC이 해결
- **따라서**: Recall만으로 충분

---

## 💭 남은 질문

### Q1: LP_r의 비선형성은 충분한가?

**현재**:
- threshold 이내: 1점
- threshold 초과: 0점
- Step function

**개선안**:
- Soft threshold (sigmoid, exponential)
- 거리 비례 가중치

### Q2: Threshold 값은 적절한가?

**논문**: LP₀, LP₁, LP₂, LP₃, LP₅, LP₁₀
**현재**: threshold=50px (매우 관대)

**실험 필요**:
- 다양한 threshold 테스트
- 이미지 해상도 고려 (2448×3264)

### Q3: 환경 예측 실패 문제는?

**Metric 변경으로 해결될까?**
- 아마 아닐 것
- GP의 환경 효과 학습 실패가 근본 원인
- Metric은 부차적 문제

**→ 환경 제거가 여전히 필요할 수 있음**

---

## 📊 예상 시나리오

### Scenario 1: LP_r이 성공 (낙관)

```
빠른 테스트 (10 iters):
  - CVaR-Score correlation > 0.3 ✓
  - Score 분포 개선
  - Perfect score < 30%

→ 50 iterations로 확장
→ CVaR 0.7+ 달성
→ 성공!
```

### Scenario 2: LP_r도 실패 (현실)

```
빠른 테스트:
  - CVaR-Score correlation < 0.3 ✗
  - 여전히 문제

→ 근본 원인: 환경 예측 실패
→ 환경 제거 실험
→ Baseline 확보
```

### Scenario 3: 중간 개선 (가능성)

```
빠른 테스트:
  - Correlation 약간 개선 (0.1 → 0.2)
  - 하지만 여전히 부족

→ Threshold 조정 실험
→ 또는 환경 Top 2로 축소
→ 점진적 개선
```

---

## 🎓 결론

### 핵심 발견

1. **기존 LP metric은 잘못 구현됨**
   - Precision 계산 버그
   - F1 score 사용 (논문에 없음)

2. **AirLine 논문 원본은 LP_r (Recall only)**
   - GT coverage 측정
   - Precision/F1 없음

3. **우리 시스템에 적합한 이유**
   - RANSAC이 단일 선 선택
   - Over-detection 문제 없음
   - 1:1 대응 구조

### 다음 액션

1. ✅ **LP_r 구현 완료**
2. ⏳ **빠른 테스트 실행** (10 iters)
3. ⏳ **결과 분석 및 다음 전략 결정**

### 기대

**낙관적**: LP_r이 문제 해결, CVaR 0.7+ 달성
**현실적**: 약간 개선, 환경 제거 여전히 필요
**비관적**: 변화 없음, 근본 원인은 환경 예측 실패

---

**마지막 업데이트**: 2025-11-16
**상태**: LP_r 구현 완료, 테스트 대기
**다음**: 빠른 실험으로 검증

**"The devil is in the details - 논문 원본을 확인하라!"** 🔍
