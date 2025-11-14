# 🔥 다음 세션 가이드 - 2025-11-14 (세션 12 준비)

**상황**: ✅ Top 6 환경 특징 선택 완료! ⚠️ 100개 평가 완료 (0.0 점수 이슈 발견)
**환경**: Windows 로컬
**Python**: `/c/Users/user/.conda/envs/weld2024_mk2/python.exe`

---

## 🎯 **현재 완료 상태 (세션 11 - 완료)**

### ✅ Multi-ROI 전략 실험 완료

**3가지 전략 비교:**
1. ✅ first_only (baseline) - 첫 번째 ROI만
2. ✅ average - 모든 ROI 평균
3. ✅ worst_case - 각 특징별 최악값 ⭐️ **WINNER!**

**결과:**
- **worst_case 전략이 최고** (|r| = 0.42)
- 상관관계 +41.9% 개선 (0.296 → 0.420)
- MODERATE → **STRONG** correlation

### ✅ PSNR/SSIM Quality Metrics 실험 완료

**추가 특징:**
- psnr_score (Peak Signal-to-Noise Ratio)
- ssim_score (Structural Similarity Index)

**결과:**
- PSNR/SSIM: **도움 안 됨** (r < 0.1)
- Gaussian blur reference가 용접 난이도 포착 실패
- 결론: 제외!

### ✅ Top 6 특징 선택 완료

**선정 기준**: 상관관계 강도 (|r| > 0.2)

**최종 6D 환경 벡터:**
1. **local_contrast**: -0.42 (STRONG) ⭐️
2. **clip_rough**: 0.40 (STRONG) ⭐️
3. **brightness**: 0.22 (MODERATE)
4. **clip_smooth**: 0.21 (MODERATE)
5. **gradient_strength**: -0.21 (MODERATE)
6. **edge_density**: 0.20 (WEAK)

**파일:**
- ✅ `environment_top6.json` (113 images, 6D)

**문서:**
- ✅ `MULTI_ROI_STRATEGY_RESULTS.md`
- ✅ `FEATURE_SELECTION_RESULTS.md`

---

## 📊 **핵심 인사이트**

### 왜 worst_case가 최고인가?

**정보량 비교:**
```
first_only (1 ROI) < average (평균) < worst_case (최악 포착)
   r = 0.296          r = 0.364        r = 0.420 ⬆️
```

**이유:**
1. **BoRisk 철학 일치** - worst α% 고려
2. **극단적 난이도 포착** - 가장 어려운 ROI 반영
3. **변별력 향상** - 난이도 차이 명확히 구분

### 특징 중요도

**Baseline vs CLIP:**
- Baseline의 `local_contrast`가 최강 (-0.42)
- CLIP의 `clip_rough`가 2위 (0.40)
- 둘 다 STRONG → **조합이 중요!**

**전략에 따른 변화:**
| 특징 | first_only | average | worst_case |
|------|-----------|---------|------------|
| local_contrast | -0.23 | -0.36 | **-0.42** ⬆️ |
| clip_rough | 0.25 | 0.35 | **0.40** ⬆️ |

→ worst_case가 진짜 난이도를 포착!

---

---

## 🚀 **다음 세션 할 일 (Priority 순서) - Session 12**

### Priority 0: 0.0 점수 이슈 디버깅 (긴급! 🚨)

**현재 상황:**
- 100개 이미지 모두 score = 0.0
- 평가는 성공했으나 점수 계산 문제
- 통계 분석 불가능

**디버깅 방법:**

**1단계: 개별 로그 확인**
```bash
cat logs_random/iter_001.json
cat logs_random/iter_050.json
# score 필드 확인
```

**2단계: GT 데이터 확인**
```bash
python -c "import json; print(list(json.load(open('../dataset/ground_truth.json')).items())[0])"
# GT 구조 확인
```

**3단계: 파이프라인 테스트**
```python
# test_evaluation.py 작성
# 단일 이미지로 전체 파이프라인 테스트
# 각 단계별 출력 확인:
# 1. YOLO 검출 결과
# 2. AirLine 검출 좌표
# 3. GT 좌표
# 4. line_equation_evaluation 점수
```

**4단계: 파라미터 조정**
```python
# Default 파라미터가 너무 엄격할 수 있음
# 더 느슨한 파라미터로 테스트:
LOOSE_PARAMS = {
    'edgeThresh1': -5.0,
    'simThresh1': 0.90,
    'pixelRatio1': 0.08,
    'edgeThresh2': 3.0,
    'simThresh2': 0.70,
    'pixelRatio2': 0.08
}
```

**예상 소요**: 1-2시간
**중요도**: 🔴 Critical (통계 분석 전제 조건!)

---

## 🚀 **다음 세션 할 일 (Priority 순서)**

### ✅ ~~Priority 0: 이미지 평가 개수 늘리기~~ (완료!)

**결과:**
- ✅ `evaluate_random_images.py` 작성 완료
- ✅ 100개 이미지 평가 완료 (seed=42)
- ✅ `logs_random/` 디렉토리 생성
- ⚠️ **문제 발견**: 모든 점수가 0.0

**이슈:**
- 100개 모두 score = 0.0 (mean, std, min, max 전부 0.0)
- 원인 불명 (default 파라미터 문제? 파이프라인 오류?)
- 다음 세션에서 디버깅 필요

---

### Priority 1: BoRisk 실험 (Top 6 특징 사용)

**목표**: 6D 환경 벡터로 CVaR 최적화 성능 검증

**1단계: optimization.py 수정**
```python
# 환경 파일 변경
ENV_FILE = "environment_top6.json"  # 13D → 6D

# w_dim 변경
w_dim = 6  # 13 → 6

# GP 입력 차원
input_dim = param_dim + w_dim  # 9 + 6 = 15D
```

**2단계: 실험 실행**
```bash
# 30 iterations with 6D environment
python optimization.py \
    --iterations 30 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_top6.json
```

**3단계: 결과 분석**
- CVaR 개선 확인
- 13D vs 6D 비교
- 수렴 속도 비교

**예상 소요**: 2-3시간 (실험 시간 포함)
**중요도**: 🔴 High (핵심 실험!)

---

### Priority 2: 자동 라벨링 시스템 (아직 미완성)

**목표**: AirLine 결과로 GT 자동 생성

**작업:**
```bash
# auto_labeling.py 작성
python auto_labeling.py \
    --image_dir ../dataset/images/test \
    --output ../dataset/ground_truth_auto.json
```

**예상 소요**: 1시간
**중요도**: 🟡 Medium

---

### Priority 3: 시각화 및 논문 Figure 생성

**목표**: 논문용 결과 Figure 생성

**작업:**
1. **상관관계 히트맵**
   - 15D 전체 vs Top 6
   - 전략별 비교 (first_only, average, worst_case)

2. **CVaR 개선 추이**
   - Iteration별 CVaR 변화
   - Alpha 비교 (0.1, 0.2, 0.3, 0.4, 0.5)

3. **환경 분포**
   - Top 6 특징의 히스토그램
   - 난이도별 이미지 샘플

**예상 소요**: 2-3시간
**중요도**: 🟡 Medium (논문 준비)

---

### Priority 4: 문서 정리

**작업:**
- ✅ `NEXT_SESSION.md` (이 파일)
- ⏳ `TRACKING.md` 업데이트
- ⏳ `README.md` 업데이트 (전체 워크플로우)

**예상 소요**: 30분
**중요도**: 🟢 Low (정리 작업)

---

## 📁 **현재 파일 상태**

### 환경 데이터 (JSON)
```
environment_roi_v2.json              - 13D (worst_case, v1)
environment_roi_first_only.json      - 13D (v1)
environment_roi_average.json         - 13D (v1)
environment_roi_worst_case.json      - 13D (v1)

environment_roi_first_only_v2.json   - 15D (with PSNR/SSIM) ✅
environment_roi_average_v2.json      - 15D (with PSNR/SSIM) ✅
environment_roi_worst_case_v2.json   - 15D (with PSNR/SSIM) ✅

environment_top6.json                - ✅ 6D (SELECTED!) ⭐️

logs_random/                         - ✅ 100 images (all 0.0 scores) ⚠️
```

### 스크립트
```
extract_environment_roi.py                  - v1 (13D)
extract_environment_multi_roi.py            - ✅ v1 (13D, 3 strategies)
extract_environment_multi_roi_v2.py         - ✅ v2 (15D, 3 strategies)
environment_with_quality_metrics.py         - ✅ PSNR/SSIM utilities
create_environment_top6.py                  - ✅ Top 6 extraction
evaluate_random_images.py                   - ✅ Random evaluation (needs debug!)

analyze_clip_correlation.py                 - Correlation analysis
optimization.py                             - BoRisk main (needs w_dim update!)
```

### 문서
```
MULTI_ROI_STRATEGY_RESULTS.md          - ✅ Multi-ROI 실험 결과
FEATURE_SELECTION_RESULTS.md           - ✅ Top 6 선택 결과
SESSION_11_SUMMARY.md                  - ✅ 세션 11 전체 요약
NEXT_SESSION.md                        - ✅ 이 파일 (세션 12용 업데이트)
TRACKING.md                            - 진행 상황 (업데이트 필요)
```

---

## 💡 **다음 세션 시작 시 (Session 12)**

### 1. 먼저 읽을 것
- ✅ `SESSION_11_SUMMARY.md` ⭐️ (세션 11 전체 요약)
- ✅ `NEXT_SESSION.md` (이 파일)
- ✅ `FEATURE_SELECTION_RESULTS.md`

### 2. 먼저 할 것 (순서대로!)

**Step 1: 0.0 점수 디버깅** (필수! 🚨)
```bash
# 개별 로그 확인
cat logs_random/iter_001.json
cat logs_random/summary.json

# GT 구조 확인
python -c "import json; print(list(json.load(open('../dataset/ground_truth.json')).items())[0])"

# 테스트 스크립트 작성
# test_evaluation.py - 단일 이미지로 파이프라인 전체 테스트
```

**Step 2: 디버깅 성공 후 - 재평가 또는 BoRisk 실험**
```bash
# Option A: 파라미터 수정 후 재평가
python evaluate_random_images.py --n_images 100

# Option B: 평가 없이 바로 BoRisk 실험
python optimization.py --iterations 30 --env_file environment_top6.json
```

### 3. 결정할 것
- 0.0 점수 원인이 무엇인가?
- 재평가가 필요한가, 아니면 BO 로그의 44개만 사용할까?
- Alpha/n_w 실험을 추가로 할지?

---

## 🎯 **성공 기준 (Session 12)**

### Minimum (최소한 달성)
- [ ] 0.0 점수 이슈 해결 (디버깅)
- [ ] Top 6로 BoRisk 실험 1회 완료
- [ ] CVaR 개선 확인 (baseline 대비)
- [ ] 결과 정리 (JSON, 로그)

### Target (목표)
- [ ] 100개 이미지로 상관관계 재검증 (0.0 이슈 해결 시)
- [ ] 13D vs 6D 성능 비교
- [ ] 시각화 1-2개 완성
- [ ] 평가 안정성 확인

### Stretch (추가 목표)
- [ ] Alpha 실험 (0.1, 0.2, 0.3, 0.4, 0.5)
- [ ] n_w 실험 (10, 15, 20, 30) - 6D에 최적인 값 찾기
- [ ] 자동 라벨링 완성
- [ ] 논문 Figure 전부 완성

---

## 📝 **메모**

### 중요한 발견 (Session 11)
1. **worst_case가 압도적으로 우수** (r=0.42, +41.9% 개선)
2. **local_contrast가 최강 특징** (-0.42, STRONG)
3. **CLIP도 중요** (clip_rough 2위, 0.40)
4. **PSNR/SSIM은 실패** (r<0.1, 용접에 부적합)
5. **6D 환경 벡터 확정** (54% 차원 감소)

### 주의사항 (Session 12)
1. **100개 평가 완료했으나 모두 0.0** → 디버깅 필수! 🚨
2. **optimization.py 수정 필요** (w_dim = 6)
3. **clip_rough의 양의 상관관계** → 해석 주의 필요
4. **default 파라미터 검증 필요** → 0.0 원인일 수 있음

### 다음 논문 작성 시
- Multi-ROI worst_case 전략 → 새로운 기여
- 6D 환경 벡터 → 효율성 + 성능
- BoRisk + 환경 적응 → 강건성 확보

---

**마지막 업데이트**: 2025-11-14 세션 12 완료 (세션 13 준비)
**세션 11 성과**: Multi-ROI 전략, Top 6 선택, 100개 평가
**세션 12 성과**: 6D BoRisk 50회 완료, Resume 기능 구현, 시각화 완료
**다음 작업**: 중단 원인 파악 → Resume으로 100회 추가 OR 새 실험
**최종 목표**: Top 6로 CVaR 최적화 성능 입증 (150회)!

**🔥 50회 완료! Resume 기능 구현됨! 이제 100회 추가만 하면 끝! 🔥**
