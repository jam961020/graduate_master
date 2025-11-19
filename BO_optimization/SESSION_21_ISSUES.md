# Session 21 Issues Summary
**Date**: 2025-11-18
**Status**: 디버깅 진행 중

---

## 🔴 Critical Issues

### 1. Score 0 문제 - 선 검출됐는데 score 0
**증상**:
- iter_001, iter_002에서 선 3개 검출됨
- GT와 어느 정도 대응됨 (시각적으로 확인)
- 그런데 score = 0.0000

**가능한 원인**:
- LP_r threshold (30px)가 너무 엄격
- 검출된 선과 GT 사이 거리가 30px 초과
- 또는 detected_coords 반환 문제

**확인 필요**:
```python
# evaluation.py line ~110
pixel_scores = np.clip(1.0 - min_distances / threshold, 0.0, 1.0)
```
- threshold=30px로 설정됨
- 거리가 30px 이상이면 score 0

**해결책**:
- [ ] threshold를 50px 또는 100px로 완화
- [ ] 또는 거리 비례 점수 방식 개선
- [ ] detected_coords가 제대로 반환되는지 확인

---

### 2. 200번대/16:28 시간대 이미지 실패
**증상**:
- idx 277, 279, 282, 285, 296, 305 등 200번대 이미지에서 계속 실패
- 모두 2025-07-17 16:28~16:29 촬영 이미지
- KG가 이 영역을 계속 탐험하려고 시도 → CVaR 하락

**원인**:
- 해당 시간대 이미지들이 검출하기 어려운 특성 보유
- 또는 GT 라벨이 잘못됨
- Sobol 샘플링이 환경 공간에서 그 영역을 계속 커버

**해결책**:
- [ ] 해당 이미지들 직접 확인 (GT 정확성 검증)
- [ ] 문제 이미지 제외 (ground_truth_auto.json에서 제거)
- [ ] 또는 score 0에 음수 페널티 부여 (GP가 회피 학습)

---

### 3. Environment JSON 부족 (222개 누락)
**현재 상태**:
- `environment_top6.json`: 113개 이미지만 포함
- `ground_truth_auto.json`: 335개 이미지
- **222개 이미지**가 on-the-fly 추출 필요 → 느리고 불안정

**해결책**:
- [ ] `environment_335.json` 생성 (335개 전체)
- NEXT_SESSION.md에 스크립트 예시 있음

---

## 🟡 Structural Issues

### 4. KG가 현재 w_set에서만 선택
**구조**:
```
매 iteration:
1. w_set 새로 샘플링 (seed=iteration)
2. 15개 환경에서 판타지 관측
3. 그 중에서 CVaR 개선 최대인 (x, w_idx) 선택
```

**문제**:
- w_set 밖의 환경은 선택 불가
- 비슷한 환경이 계속 선택될 수 있음

**해결책**:
- BoRisk 논문 설계대로임 (변경 어려움)
- 대신 문제 이미지 제외로 우회

---

## ✅ Completed

### 시각화 저장 기능 추가
- `debug_visualizer.py` 생성
- `optimization.py`에 통합 완료
- 매 iteration마다 이미지 저장
- YOLO bbox, GT (초록), Detected (빨강) 표시

### 선 추출 로직 수정
- 점 6개 → 선 5개 구조 반영
- longi_left, longi_right, collar_left 세로선
- longi_left_lower-collar_left_lower, collar_left_lower-longi_right_lower 가로선

---

## 📋 Next Session TODO

### Priority 1: Score 0 원인 파악
1. evaluation.py의 LP_r 로직 확인
2. threshold 완화 테스트 (30 → 50 또는 100)
3. 실패 이미지의 detected_coords 디버깅

### Priority 2: 문제 이미지 처리
1. 16:28 시간대 이미지 GT 검증
2. 문제 있으면 ground_truth_auto.json에서 제거
3. 또는 score 0에 -0.5 페널티 부여

### Priority 3: Environment JSON 생성
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
python -c "
from environment_independent import extract_parameter_independent_environment
from pathlib import Path
import json, cv2

image_dir = Path('../dataset/images/test')
images = sorted(list(image_dir.glob('*.jpg')))

all_env = {}
for i, img_path in enumerate(images):
    if i % 50 == 0:
        print(f'Progress: {i}/{len(images)}')
    img = cv2.imread(str(img_path))
    env = extract_parameter_independent_environment(img, None)
    all_env[img_path.stem] = env

with open('environment_335.json', 'w') as f:
    json.dump(all_env, f, indent=2)
print(f'Saved: environment_335.json ({len(all_env)} images)')
"
```

### Priority 4: 본 실험 실행
```bash
python optimization.py \
  --iterations 50 \
  --n_initial 5 \
  --alpha 0.3 \
  --n_w 10 \
  --gt_file ../dataset/ground_truth_auto.json \
  --env_file environment_335.json
```

---

## 📁 Debug Images Location

```
logs/run_20251118_175735/debug_images/
├── iter_001_FAIL_s0p000_WIN_20250717_16_28_26_Pro.jpg
├── iter_002_FAIL_s0p000_WIN_20250717_16_28_50_Pro.jpg
├── iter_003_s0p797_WIN_20250605_10_47_30_Pro.jpg
```

---

## 🔍 Key Observations

1. **선이 검출됐는데 score 0**: LP_r threshold 문제 의심
2. **16:28 시간대 집중 실패**: 특정 촬영 조건 문제 또는 GT 오류
3. **CVaR 0.0000 시작**: 초기 샘플에서 score 0 다수 발생
4. **KG 정상 작동**: 실패 영역 탐험하는 것은 알고리즘적으로 합리적

---

**작성일**: 2025-11-18 18:05
**다음 세션**: Score 0 원인 파악부터 시작
