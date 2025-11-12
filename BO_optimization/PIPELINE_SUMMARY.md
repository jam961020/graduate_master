# 전체 파이프라인 요약 (2025-11-13)

## 🎯 핵심 질문: "왜 RANSAC을 따로 만들었나?"

**답변**: **BO 최적화를 위해!**

---

## 파이프라인 비교

### AirLine 원본 vs full_pipeline 커스텀

#### 1. AirLine_assemble_test.py `find_best_fit_line_ransac()`
```python
# 픽셀 복제 방식 (고정 가중치)
pixel_pool.extend(list(other_only) * 1)       # 다른 알고리즘: 1회
pixel_pool.extend(list(one_air_only) * 3)     # AirLine 하나: 3회
pixel_pool.extend(list(one_air_and_other) * 5)  # 겹침: 5회
pixel_pool.extend(list(both_air) * 10)         # 둘 다 AirLine: 10회

# sklearn RANSACRegressor
ransac = RANSACRegressor(max_trials=10000)
ransac.fit(X, y)
```

**특징**:
- ❌ **가중치 하드코딩** (3, 5, 10 고정)
- ❌ **BO로 최적화 불가**
- ✅ sklearn 사용 (안정적)
- ✅ 픽셀 기반 (정확)

#### 2. full_pipeline.py `weighted_ransac_line()`
```python
# 확률 분포 방식 (파라미터화된 가중치)
lengths = [line_len(ln) for ln in all_lines]
cweights = [center_weight(ln) for ln in all_lines]
probs = w_length * lengths + w_center * cweights  # ← BO가 최적화!

if airline_mask.any():
    probs[airline_mask] *= consensus_weight  # ← BO가 최적화!

# 확률 기반 샘플링
i1, i2 = rng.choice(len(all_lines), size=2, p=probs)
```

**특징**:
- ✅ **파라미터화된 가중치** (w_center, w_length, consensus_weight)
- ✅ **BO가 최적 가중치 자동 탐색!** ← 핵심!
- ✅ 라인 단위 (직관적)
- ❌ 수동 RANSAC (버그 가능성)

---

## 전체 파이프라인 구조

```
detect_with_full_pipeline()
├─ 1. YOLO ROI 검출
│  └─ yolo_detector.detect_rois() → ROI 좌표들
│
├─ 2. 각 ROI별 처리
│  ├─ 전처리
│  │  ├─ Grayscale + Gaussian Blur
│  │  ├─ sharp_S() ← AirLine 원본
│  │  └─ enhance_color() ← AirLine 원본
│  │
│  ├─ detect_lines_in_roi() ← 선 검출
│  │  ├─ run_lsd() ← AirLine 원본
│  │  ├─ run_fld() ← AirLine 원본
│  │  ├─ run_hough() ← AirLine 원본
│  │  ├─ run_airline(Q preset) ← AirLine 원본 ✨
│  │  └─ run_airline(QG preset) ← AirLine 원본 ✨
│  │  → lines_by_algo dict
│  │
│  └─ process_guideline_roi() / process_collar_roi()
│     ├─ weighted_ransac_line() ← **커스텀! (BO 최적화)**
│     ├─ find_upper_point() ← 커스텀
│     └─ 교점 계산 ← 커스텀
│
└─ 3. calculate_final_coordinates()
   → 12개 좌표 (GT와 비교)
```

---

## BO 최적화 파라미터 (9D)

### AirLine 파라미터 (6D) - 원본 사용
```python
# Q 프리셋
edgeThresh1: [-23.0, 7.0]   # 엣지 임계값
simThresh1: [0.5, 0.99]     # 유사도
pixelRatio1: [0.01, 0.15]   # 픽셀 비율

# QG 프리셋
edgeThresh2: [-23.0, 7.0]
simThresh2: [0.5, 0.99]
pixelRatio2: [0.01, 0.15]
```

### RANSAC 가중치 (3D) - 커스텀 구현
```python
ransac_center_w: [0.0, 1.0]    # 중심 거리 가중치
ransac_length_w: [0.0, 1.0]    # 라인 길이 가중치
ransac_consensus_w: [1, 10]    # AirLine 합의 부스팅
```

---

## 요약

1. ✅ **AirLine 알고리즘 자체는 원본 그대로 사용**
   - `run_airline()` 함수 그대로
   - Q, QG 프리셋 파라미터만 BO 최적화

2. ✅ **RANSAC만 커스텀 구현**
   - 라인 선택 가중치를 BO 최적화 가능하게
   - `find_best_fit_line_ransac` (원본) 사용 안 함!

3. ✅ **나머지 모두 AirLine 원본 활용**
   - 전처리, 선 검출, 유틸리티 함수들

4. ❌ **버그**: `weighted_ransac_line()` 라인 1개일 때 크래시
   - Line 261: early return 있음 ✅
   - Line 318: 2개 샘플링 시도 ← 왜 여기 도달? ❓

---

**작성일**: 2025-11-13 00:05
**다음 세션**: 이 파일을 먼저 읽어보세요!
