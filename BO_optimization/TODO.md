# 전체 할 일 정리 (TODO)

**업데이트**: 2025-11-14 세션 10
**현재 상태**: ROI 기반 환경 추출 완료, BoRisk 실험 준비 완료

---

## 🎯 즉시 할 일 (High Priority)

### 1. 자동 라벨링 완료 (진행 중)

**목적**: 113개 전체 이미지 자동 라벨링

**명령어**:
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
conda activate weld2024_mk2
python auto_labeling.py \
    --image_dir ../dataset/images/test \
    --output ../dataset/ground_truth_auto.json
```

**예상 시간**: 30-60분 (AirLine 속도에 따라)

**성공 기준**:
- 113개 이미지 중 90개 이상 성공
- ground_truth_auto.json 생성

---

### 2. BoRisk 최적화 실험 실행

**파일 수정 필요**: `optimization.py`

**현재 문제**:
- `optimization.py`에서 환경 벡터를 아직 사용하지 않음
- w_set 샘플링 시 `environment_roi_v2.json` 로드 필요

**수정 사항**:
```python
# optimization.py에 추가 필요

# 1. 환경 데이터 로드
import json
with open('environment_roi_v2.json') as f:
    environment_data = json.load(f)

# 2. w_set 샘플링 시 환경 사용
def sample_w_set(images_data, n_w):
    """Sample n_w environments from images"""
    sampled_images = np.random.choice(len(images_data), size=n_w, replace=False)
    w_set = []
    for idx in sampled_images:
        img_name = images_data[idx]['name']
        env_vector = environment_data[img_name]
        # 13D vector: [9D baseline + 4D CLIP]
        w = [env_vector[k] for k in sorted(env_vector.keys())]
        w_set.append(w)
    return torch.tensor(w_set, dtype=torch.float32)
```

**실험 명령어** (수정 후):
```bash
python optimization.py \
    --iterations 30 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_roi_v2.json
```

**예상 시간**: 2-4시간 (iterations에 따라)

---

### 3. Baseline 비교 실험

**목적**: 환경 벡터 효과 검증

**실험 조합**:
```bash
# A. 환경 없음 (파라미터만)
python optimization.py --iterations 20 --no_env

# B. Baseline만 (9D)
python optimization.py --iterations 20 --env_type baseline

# C. CLIP만 (4D)
python optimization.py --iterations 20 --env_type clip

# D. Baseline + CLIP (13D)
python optimization.py --iterations 20 --env_type all
```

**비교 지표**:
- 최종 CVaR
- 수렴 속도
- 안정성 (표준편차)

---

## 📊 중간 우선순위 (Medium Priority)

### 4. 시각화 생성

**필요한 그래프**:

1. **환경 벡터 분포**
   - 13D 특징의 히스토그램
   - PCA 2D 시각화

2. **상관관계 히트맵**
   - 13개 특징 vs 성능
   - 특징 간 상관관계

3. **BoRisk 실험 결과**
   - CVaR 개선 추이
   - 환경별 성능 분포
   - Alpha 비교 (0.1, 0.2, 0.3)

**스크립트 작성**:
```bash
python visualize_results.py \
    --env_file environment_roi_v2.json \
    --results_dir results/ \
    --output_dir figures/
```

---

### 5. 성능 분석 및 보고서

**분석 항목**:

1. **환경 특징 효과**
   - Baseline vs CLIP vs 조합
   - 어떤 특징이 가장 중요한가?

2. **BoRisk 개선도**
   - 환경 고려 전 vs 후
   - CVaR 개선율

3. **실패 케이스 분석**
   - 성능 낮은 이미지 특성
   - 환경 벡터 패턴

**출력**: `RESULTS.md`

---

## 🔬 낮은 우선순위 (Low Priority / 시간 있으면)

### 6. 환경 특징 추가 실험

**시도해볼 것**:

1. **다른 CLIP 프롬프트**
   ```python
   # 현재
   "a bright clear well-lit image"
   "a dark shadowy poorly-lit image"

   # 시도
   "high quality sharp image"
   "low quality blurry image"
   "high contrast image"
   "low contrast image"
   ```

2. **물리적 특징 추가**
   - Histogram equalization 전후 차이
   - Fourier transform 특징
   - HOG (Histogram of Oriented Gradients)

3. **Deep features**
   - ResNet 특징 (CNN 중간층)
   - DINO 특징 (self-supervised)

---

### 7. BoRisk 알고리즘 개선

**현재 문제** (NEXT_SESSION.md 참조):

**치명적!** 매 iteration 15개 평가 중 (1개만 해야 함!)

**수정 필요**:
```python
# borisk_kg.py
def optimize(self, ...):
    # 현재: x만 반환
    # 필요: (x, w_idx) 반환
    return best_x, best_w_idx

# optimization.py
# 현재: w_set 전부 평가
for w in w_set:
    evaluate(best_x, w)

# 필요: 선택된 w만 평가
selected_w = w_set[best_w_idx]
evaluate(best_x, selected_w)
```

**예상 효과**: 15배 속도 향상!

---

### 8. 하이퍼파라미터 튜닝

**실험할 것**:

| 파라미터 | 현재 | 시도 |
|---------|------|------|
| n_w | 15 | 10, 20, 30 |
| alpha | 0.3 | 0.1, 0.2, 0.4, 0.5 |
| num_fantasies | 64 | 32, 128 |
| n_initial | 10 | 5, 15, 20 |

**Grid search**:
```bash
for alpha in 0.1 0.2 0.3 0.4; do
    for n_w in 10 15 20; do
        python optimization.py \
            --alpha $alpha \
            --n_w $n_w \
            --iterations 20
    done
done
```

---

## 📝 문서화 (Ongoing)

### 완료된 문서:
- ✅ `ENVIRONMENT_EXTRACTION.md` - 환경 추출 가이드
- ✅ `TRACKING.md` - 작업 진행 상황
- ✅ `TODO.md` - 이 파일

### 추가 필요:
- ⏳ `RESULTS.md` - 실험 결과 정리
- ⏳ `BORISK_IMPLEMENTATION.md` - BoRisk 구현 상세
- ⏳ `VISUALIZATION_GUIDE.md` - 시각화 가이드

---

## 🎯 최종 목표 (논문용)

### 필수 실험:

1. **Baseline 비교**
   - Random Search
   - Grid Search
   - Standard BO (EI)
   - BoRisk (환경 없음)
   - BoRisk (환경 있음) ← 우리

2. **환경 벡터 ablation**
   - No environment
   - Baseline only (9D)
   - CLIP only (4D)
   - Baseline + CLIP (13D)

3. **Alpha 비교**
   - α = 0.1 (worst 10%)
   - α = 0.2 (worst 20%)
   - α = 0.3 (worst 30%)
   - α = 0.5 (median)

### 필수 Figure:

1. **Main Results**
   - CVaR 개선 추이 (methods 비교)
   - 환경별 성능 분포
   - Alpha별 수렴 곡선

2. **Ablation Study**
   - 환경 특징 효과
   - 상관관계 히트맵

3. **Qualitative Results**
   - 초기 vs 최종 검출 결과
   - 실패 케이스 분석

---

## 📅 타임라인 (추정)

| 작업 | 예상 시간 | 마감 |
|------|----------|------|
| 자동 라벨링 | 1시간 | 오늘 |
| BoRisk 실험 (1개) | 3시간 | 내일 |
| Baseline 비교 (4개) | 12시간 | 2일 |
| 시각화 | 3시간 | 2일 |
| 보고서 작성 | 4시간 | 3일 |
| **Total** | **23시간** | **3일** |

---

## ✅ 체크리스트

### 즉시 (오늘)
- [ ] 자동 라벨링 완료
- [ ] optimization.py 환경 벡터 통합
- [ ] BoRisk 첫 실험 실행

### 내일
- [ ] Baseline 비교 실험 4개
- [ ] 결과 분석
- [ ] 주요 Figure 생성

### 모레
- [ ] 추가 실험 (시간 있으면)
- [ ] 보고서 작성
- [ ] 문서 정리

---

## 💡 참고사항

### 상관관계 0.3의 의미:

| \|r\| 범위 | 강도 | 해석 |
|----------|------|------|
| 0.0 - 0.1 | NEGLIGIBLE | 거의 무관 |
| 0.1 - 0.2 | WEAK | 약한 관계 |
| **0.2 - 0.3** | **WEAK-MODERATE** | **BoRisk 사용 가능** |
| 0.3 - 0.5 | MODERATE | 중간 강도 |
| 0.5+ | STRONG | 강한 관계 |

**우리의 0.296:**
- BoRisk에 충분히 의미있음
- 환경이 성능 변동의 ~9% 설명
- 나머지 91%는 파라미터가 설명 → 괜찮음!

### 환경 벡터 사용 이유:

BoRisk는 **파라미터(x) + 환경(w)** 동시 최적화:
- 환경이 10%만 설명해도 CVaR 개선 가능
- 다양한 환경에서 robust한 파라미터 찾기
- 최악의 경우(worst α%)에서도 잘 작동하는 x

---

**마지막 업데이트**: 2025-11-14
**다음 체크**: 자동 라벨링 완료 후
