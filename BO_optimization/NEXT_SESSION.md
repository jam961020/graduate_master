# 🔥 긴급 세션 가이드 - 2025-11-14 (세션 10) - ROI 기반 CLIP 재실행!

**상황**: ⚠️ **CLIP은 작동하지만 전체 이미지로 돌림! ROI로 재실행 필요!**
**환경**: Windows 로컬
**Python**: `/c/Users/user/.conda/envs/weld2024_mk2/python.exe`

---

## 🔍 **현재 상황 (세션 9 결과)**

### ✅ 완료된 작업

1. **CLIP 설치 및 테스트** ✅
   - CLIP ViT-B/32 모델 로드 성공
   - 6D 의미적 환경 벡터 생성 확인
   
2. **CLIP 환경 인코더 구현** ✅
   - `clip_environment.py` 작성
   - 6개 용접 특화 프롬프트:
     ```python
     prompts = [
         "a clear welding ROI with good visibility",
         "a welding ROI with heavy dark shadows",
         "a welding ROI with metal debris and particles",
         "a welding ROI with bright specular reflections",
         "a welding ROI with weld beads obstructing the line",
         "a welding ROI with complex texture and noise"
     ]
     ```

3. **전체 이미지 CLIP 특징 추출** ✅
   - 113/113 이미지 처리 완료
   - `environment_clip.json` 생성됨

4. **상관관계 분석** ✅
   - CLIP vs Baseline 비교 완료

---

## ❌ **치명적 문제 발견!**

### 문제: 전체 이미지로 CLIP 돌림!

**현재 상황**:
```python
# extract_clip_features.py에서
detections = yolo_detector.detect(image)  # ← 이 메서드 없음!
# Exception 발생 → fallback으로 전체 이미지 사용
roi_crop = image  # ← 전체 이미지!
```

**결과**:
- 113개 이미지 모두 **전체 이미지**로 CLIP 인코딩
- ROI(용접 부분) 아니라 배경, 여백 전부 포함
- 상관관계 약함 (clip_beads: r = -0.177)

**왜 문제인가**:
- 용접 부분의 그림자, 철가루, 비드는 **ROI 내부**에만 존재
- 전체 이미지는 대부분 배경, 테이블, 벽 등
- CLIP이 배경 특징을 학습 → 성능과 무관

---

## 🎯 **긴급 해결 방법 (세션 10)**

### Step 1: YOLO ROI 추출 수정 (30분)

**문제 파악**:
```python
# yolo_detector.py 확인 필요
class YOLODetector:
    def detect(self, image):  # ← 이 메서드 있나?
        ...
```

**해결 방법 A**: `yolo_detector.py` 읽고 올바른 메서드명 찾기
```bash
# 예상 메서드명
- predict(image)
- infer(image)
- __call__(image)
```

**해결 방법 B**: `full_pipeline.py`에서 YOLO 사용법 확인
```python
# full_pipeline.py에서 YOLO 어떻게 쓰는지 확인
from full_pipeline import detect_with_full_pipeline
```

**수정 파일**: `extract_clip_features.py`
```python
# 수정 전
detections = yolo_detector.detect(image)  # ← 에러!

# 수정 후 (예시)
detections = yolo_detector.predict(image)  # 또는
results = yolo_detector(image)  # 또는
bbox = get_roi_from_yolo(image, yolo_detector)  # full_pipeline에서 가져오기
```

---

### Step 2: ROI 기반 CLIP 재추출 (30분)

**실행**:
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

# YOLO 수정 후 재실행
/c/Users/user/.conda/envs/weld2024_mk2/python.exe extract_clip_features.py \
    --output environment_clip_roi.json
```

**확인 사항**:
- YOLO ROI 검출 성공 (에러 없음)
- 전체 이미지 fallback 없음
- 113개 모두 ROI 기반 추출

---

### Step 3: ROI 기반 상관관계 재분석 (15분)

```bash
/c/Users/user/.conda/envs/weld2024_mk2/python.exe analyze_clip_correlation.py \
    --clip_features environment_clip_roi.json
```

**기대 결과**:
- 상관관계 증가: |r| > 0.3 기대
- `clip_shadow`, `clip_debris`, `clip_beads` 등이 유의미하게

---

## 📋 **체크리스트 (순차 실행)**

### ✅ Priority 0: YOLO ROI 추출 수정

```bash
# 1. yolo_detector.py 확인
cat yolo_detector.py | grep "def "

# 2. full_pipeline.py에서 YOLO 사용법 확인
grep -A 10 "yolo_detector" full_pipeline.py | head -20

# 3. extract_clip_features.py 수정
# - 올바른 YOLO 메서드 사용
# - ROI 추출 로직 검증
```

### ✅ Priority 1: ROI 기반 CLIP 재추출

```bash
# 재실행 (30분 소요)
python extract_clip_features.py --output environment_clip_roi.json

# 확인
ls -lh environment_clip_roi.json
# 113개 이미지 모두 포함 확인
```

### ✅ Priority 2: ROI 상관관계 분석

```bash
# 분석 (5분)
python analyze_clip_correlation.py --clip_features environment_clip_roi.json

# 기대: |r| > 0.3
```

### ✅ Priority 3: BoRisk with ROI-CLIP (조건부)

**조건**: 상관관계 |r| > 0.25 이상인 경우만

```bash
# optimization.py 수정 후 실행
python optimization.py \
    --iterations 30 \
    --n_initial 5 \
    --alpha 0.1 \
    --n_w 15 \
    --env_type clip \
    --clip_features environment_clip_roi.json
```

---

## 🔧 **YOLO ROI 추출 디버깅 가이드**

### 방법 1: yolo_detector.py 직접 확인

```python
# Read yolo_detector.py
from yolo_detector import YOLODetector

detector = YOLODetector("models/best.pt")

# 메서드 확인
print(dir(detector))

# 테스트
import cv2
img = cv2.imread("../dataset/images/test/WIN_20250604_14_01_48_Pro.jpg")
result = detector.predict(img)  # 또는 다른 메서드
print(result)
```

### 방법 2: full_pipeline.py에서 ROI 추출 로직 복사

```python
# full_pipeline.py 160번째 줄 근처
def detect_with_full_pipeline(image, params, yolo_detector, ransac_weights):
    # ROI 추출 부분 찾기
    # 해당 로직을 extract_clip_features.py에 복사
```

### 방법 3: Fallback - YOLO 없이 고정 ROI 사용

```python
# 만약 YOLO 안 되면 고정 ROI 사용
def get_fixed_roi(image):
    h, w = image.shape[:2]
    # 중앙 60% 영역 사용
    x1 = int(w * 0.2)
    y1 = int(h * 0.2)
    x2 = int(w * 0.8)
    y2 = int(h * 0.8)
    return (x1, y1, x2, y2)
```

---

## 📊 **현재 vs 기대 상관관계**

### 현재 (전체 이미지 CLIP)

| Feature | Correlation | Strength |
|---------|-------------|----------|
| clip_beads | -0.177 | WEAK |
| clip_shadow | 0.065 | NEGLIGIBLE |
| 기타 | < 0.05 | NEGLIGIBLE |

**Baseline** (brightness, contrast, etc.): r = -0.135

**개선률**: +31% (0.177 vs 0.135)

### 기대 (ROI 기반 CLIP)

| Feature | Expected | Reasoning |
|---------|----------|-----------|
| clip_shadow | > 0.3 | ROI 내 그림자는 선 검출 방해 |
| clip_debris | > 0.25 | 철가루는 ROI에만 |
| clip_beads | > 0.3 | 용접 비드는 ROI 특화 |
| clip_reflection | > 0.2 | 금속 반사는 ROI 중심 |

**예상 개선률**: +100~200% (0.3~0.4 vs 0.135)

---

## 💡 **핵심 인사이트**

### 왜 ROI가 중요한가?

**전체 이미지 문제**:
```
[전체 이미지]
┌─────────────────────────┐
│                         │
│   테이블, 벽, 배경      │ ← CLIP이 이것 학습
│                         │
│    ┌──────────┐         │
│    │ ROI 영역 │         │ ← 우리가 관심있는 부분 (10%)
│    │ (용접)   │         │
│    └──────────┘         │
│                         │
└─────────────────────────┘
```

**ROI만 사용**:
```
[ROI 크롭]
┌──────────┐
│          │
│ 용접선   │ ← CLIP이 이것만 학습
│ 그림자   │
│ 철가루   │
│ 비드     │
│          │
└──────────┘
```

### 성능 영향

- **그림자**: 어두운 ROI → Canny 엣지 약함 → 선 검출 실패
- **철가루**: 노이즈 많음 → RANSAC 방해
- **용접 비드**: 타원형 blob → 직선 방해
- **금속 반사**: 과도한 밝기 → 엣지 손실

→ 이런 특징들은 **ROI 내부**에만 존재!

---

## ⚠️ **중요 메모**

### 다음 세션 시작 시

1. **yolo_detector.py 먼저 확인!**
   ```bash
   cat yolo_detector.py | grep "class\|def"
   ```

2. **full_pipeline.py에서 ROI 추출 방법 확인**
   ```bash
   grep -A 20 "yolo_detector" full_pipeline.py
   ```

3. **extract_clip_features.py 수정 후 재실행**

4. **상관관계 재분석 → 0.3 이상 나오면 BoRisk 실행**

---

## 📁 **생성된 파일**

### 완료
- ✅ `clip_environment.py` - CLIP 인코더 클래스
- ✅ `extract_clip_features.py` - 특징 추출 스크립트 (수정 필요!)
- ✅ `analyze_clip_correlation.py` - 상관관계 분석
- ✅ `environment_clip.json` - 전체 이미지 CLIP (잘못됨!)

### 다음 생성 필요
- ⏳ `environment_clip_roi.json` - **ROI 기반 CLIP** (목표!)

---

## 🎯 **성공 기준**

1. **YOLO ROI 추출 성공** (에러 없이)
2. **ROI 기반 CLIP 상관관계**: |r| > 0.25
3. **Baseline 대비 2배 이상 개선**: 0.3 vs 0.135

만약 안 되면:
- Plan B: 용접 특화 physical features (shadow, reflection)
- Plan C: 환경 무시, 파라미터만 최적화

---

**마지막 업데이트**: 2025-11-14 세션 9
**다음 작업**: YOLO ROI 추출 수정 → ROI 기반 CLIP 재실행!
**목표**: ROI 기반으로 상관관계 0.3 이상 달성!

**🔥 ROI만 보면 확실히 상관관계 올라갈 거야! 화이팅! 🔥**
