# 작업 진행 상황 트래킹

**프로젝트:** BoRisk CVaR Optimization for Welding Line Detection
**시작일:** 2025.11.11
**환경:** Windows 로컬 (Linux segfault로 회귀)

---

## 🔥 최신 업데이트 (2025.11.13)

### ✅ Full BoRisk-KG 판타지 관측 활성화 (2025.11.13 02:30)

**문제:**
- `posterior.rsample()`이 `[1, n_w, 1]` shape 반환
- `train_Y`는 `[N]` shape (1D)
- torch.cat 실패: "Tensors must have same number of dimensions"

**해결:**
- `fantasy_obs`를 `squeeze()`로 1D 변환
- `_create_fantasy_model`에서 dimension 체크 추가

**결과:**
- ✅ Full BoRisk-KG 정상 작동
- ✅ 16개 판타지 관측 생성 확인
- ✅ "Using BoRisk-KG" 출력 (이전: "Simplified-CVaR-KG")

**파일:** `borisk_kg.py:90-111`, `optimization.py:699-701`

---

### ✅ KG current_best_cvar 버그 수정 (2025.11.13 02:50)

**문제**:
- `_compute_current_best_cvar()`가 **마지막 x의 CVaR만** 계산
- 이전 평가된 x들 완전 무시
- **KG가 엉뚱한 지점 선택** → 개선 안 됨!

**수정**:
```python
# Before: 마지막 것만
last_scores = train_Y[-self.n_w:]
current_cvar = worst_scores.mean().item()

# After: 모든 x에 대해 CVaR 계산 후 최대값
for i in range(n_groups):
    group_scores = train_Y[i*n_w:(i+1)*n_w]
    cvar = compute_cvar(group_scores)
    best_cvar = max(best_cvar, cvar)  # 최대값 선택
```

**결과**:
- ✅ 5 iterations 테스트: **+25.9% 개선!**
- ✅ Iter 5에서 CVaR 0.4609 → 0.5023
- ✅ KG가 드디어 개선 지점 선택!

**파일**: `borisk_kg.py:44-74`

---

### ✅ 자동 라벨링 스크립트 완성 (2025.11.13 03:00)

**목표**: AirLine으로 6개 점 자동 추출

**구현**:
- `detect_with_full_pipeline()` 사용
- ground_truth.json과 동일한 포맷 저장
- 유효성 검사: 최소 4개 점 검출 확인

**사용법**:
```bash
# 전체 이미지 자동 라벨링
python auto_labeling.py

# 테스트 (10개만)
python auto_labeling.py --max_images 10

# 결과: ../dataset/ground_truth_auto.json
```

**파일**: `auto_labeling.py`

---

### ✅ 평가 Metric 개선: 기울기 차이 기반 (2025.11.13 02:40)

**문제 발견:**
- 이전: 코사인 유사도 사용 `abs(A_gt*A_det + B_gt*B_det)`
- 작은 각도 차이에 둔감: cos(1°) ≈ 0.9998, cos(5°) ≈ 0.9962
- **CVaR이 0.99+ 로 너무 높음** → 구분력 부족

**변경 내용:**
```python
# 이전 (코사인 유사도)
direction_sim = abs(A_gt*A_det + B_gt*B_det)

# 현재 (기울기 차이)
slope_gt = -A_gt / B_gt if abs(B_gt) > 1e-6 else 1e6
slope_det = -A_det / B_det if abs(B_det) > 1e-6 else 1e6
slope_diff = abs(slope_gt - slope_det)
direction_sim = 1.0 / (1.0 + slope_diff)
```

**Metric 구성 (optimization.py:43-119):**
1. **방향 유사도 (60%)**: 기울기 차이 기반
   - slope = -A/B (직선 방정식 Ax + By + C = 0)
   - 정규화: `1/(1 + |slope_diff|)`
   - 기울기 차이 0 → 1.0, 0.5 차이 → 0.67, 1.0 차이 → 0.5

2. **평행 거리 (40%)**: GT 중점에서 검출 직선까지
   - GT 중점: `(mid_x, mid_y) = ((x1+x2)/2, (y1+y2)/2)`
   - 수직 거리: `|A_det*mid_x + B_det*mid_y + C_det|`
   - threshold: 대각선의 5%
   - 정규화: `max(0, 1 - dist/threshold)`

**결과:**
- ✅ CVaR: **0.99+ → 0.39~0.47** (현실적!)
- ✅ 기울기 틀어짐에 훨씬 민감
- ✅ 10 iterations 테스트에서 일관된 범위 유지

**파일:** `optimization.py:43-119`

---

## 📋 현재 우선순위

1. ✅ **CRG311 Linux 빌드 설치** (완료) - 2025.11.11 19:00
2. ✅ **평가 메트릭 변경** (완료) - 직선 방정식 기반으로 전환 2025.11.11 19:28
3. ✅ **RANSAC 가중치 추가** (완료) - 6D → 9D 확장 2025.11.11 19:28
4. ✅ **로깅 최적화** (완료) - 토큰 절약, 파일 저장 2025.11.11 19:28
5. 🔄 **테스트 실행 중** (진행중) - 수정된 코드 검증
6. 🟡 **환경 특징 강화** (대기) - CLIP, PSNR/SSIM 추가
7. 🔴 **판타지 관측 구현** (대기) - BoRisk 알고리즘
8. 🟢 **환경 변수 통합** (대기) - GP에 (x, z) 입력

---

## 🔍 발견된 문제점 상세

### 1. 환경 변수 미사용 ❌

**위치:** `optimization.py`

**문제:**
- `environment_independent.py`에 6D 환경 벡터 추출 코드 있음:
  - brightness, contrast, edge_density, texture_complexity, blur_level, noise_level
- **하지만 optimization.py에서 전혀 사용하지 않음**
- `BOUNDS`는 6D만 정의 (AirLine 파라미터만)
- GP 학습 시: `SingleTaskGP(X, Y)` → X는 [N, 6]
- 이미지별 환경을 고려하지 않음

**현재 구조:**
```python
# optimization.py:32-35
BOUNDS = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01],  # 6D
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15]
])

# optimization.py:219
def objective_function(X, images_data, ...):  # X: [1, 6]
    params = {...}  # 6개 파라미터만
    for img_data in images_data:
        score = evaluate(image, params)  # 환경 z 없음!
    return cvar(scores)
```

**BoRisk에서 필요한 것:**
```python
# 각 이미지마다 환경 z_i 추출
env_features = extract_environment(image)  # [6D]

# GP 입력: [x, z]
X_with_env = torch.cat([X, env_tensor], dim=-1)  # [N, 12D]
gp = SingleTaskGP(X_with_env, Y)

# 평가: (x, z) → y
def objective_function(X, env_z, image):
    score = evaluate(image, X)
    return score
```

**해결 방안:**
- BOUNDS를 12D 또는 15D로 확장 (params 6D + env 6~9D)
- 각 이미지 평가 시 환경 벡터 z 추출
- GP를 (x, z) → y로 학습
- 새로운 이미지 z*에서 최적 x* 예측

---

### 2. RANSAC 가중치 미연결 ⚠️

**위치:** `full_pipeline.py:330-337`, `optimization.py:233-240`

**문제:**
- `full_pipeline.py`에 RANSAC 가중치를 받는 코드는 존재:
```python
# full_pipeline.py:330-337
w_center = float(params.get('ransac_center_w', 0.5))
w_length = float(params.get('ransac_length_w', 0.5))
w_consensus = int(params.get('ransac_consensus_w', 5))
```

- **하지만 optimization.py에서 이 키를 전달하지 않음:**
```python
# optimization.py:233-240
params = {
    'edgeThresh1': X[0, 0].item(),
    'simThresh1': X[0, 1].item(),
    'pixelRatio1': X[0, 2].item(),
    'edgeThresh2': X[0, 3].item(),
    'simThresh2': X[0, 4].item(),
    'pixelRatio2': X[0, 5].item(),
    # RANSAC 가중치 없음!
}
```

**결과:** RANSAC 가중치가 항상 기본값(0.5, 0.5, 5)으로 고정

**해결 방안:**
```python
# 1. BOUNDS 확장 (6D → 9D)
BOUNDS = torch.tensor([
    [-23.0, 0.5, 0.01, -23.0, 0.5, 0.01, 0.0, 0.0, 1],
    [7.0, 0.99, 0.15, 7.0, 0.99, 0.15, 1.0, 1.0, 10]
])

# 2. params에 추가
params = {
    'edgeThresh1': X[0, 0].item(),
    'simThresh1': X[0, 1].item(),
    'pixelRatio1': X[0, 2].item(),
    'edgeThresh2': X[0, 3].item(),
    'simThresh2': X[0, 4].item(),
    'pixelRatio2': X[0, 5].item(),
    'ransac_center_w': X[0, 6].item(),   # 추가
    'ransac_length_w': X[0, 7].item(),   # 추가
    'ransac_consensus_w': int(X[0, 8].item())  # 추가
}
```

---

### 3. 판타지 관측 미구현 ❌

**위치:** `optimization.py:292-435`

**문제:** **BoRisk의 핵심 알고리즘이 완전히 누락됨**

**현재 구현:**
- 일반적인 Vanilla BO 구조
- 목적함수: `f(x) = CVaR over all images`
- 획득함수: UCB 또는 EI
- 한 iteration에 한 x를 평가하고 전체 이미지에 대한 CVaR 계산

**BoRisk에서 필요한 것:**
1. **각 이미지마다 (x, z_i) → y_i 관측**
2. **GP는 (x, z) → y 매핑 학습**
3. **새로운 환경 z*에서 최적 x* 예측 (fantasy observation)**
4. **CVaR Knowledge Gradient 획득함수 사용**

**구현 방향:**
```python
# 1. 이미지별 환경 추출 및 저장
for img_data in images_data:
    img_data['env'] = extract_environment(img_data['image'])

# 2. 평가 시 (x, z, y) 튜플 저장
observations = []
for img in sample_images:
    X_z = torch.cat([X, img['env']], dim=-1)
    y = evaluate(img['image'], X)
    observations.append((X_z, y))

# 3. GP 학습
X_train = torch.stack([obs[0] for obs in observations])
Y_train = torch.tensor([obs[1] for obs in observations])
gp = SingleTaskGP(X_train, Y_train)

# 4. CVaR-KG 획득함수
# 각 candidate x에 대해:
#   - 모든 환경 z에서 성능 예측
#   - CVaR 계산
#   - Knowledge Gradient 계산
```

---

### 4. 평가 메트릭 문제 🔧

**위치:** `optimization.py:38-117`

**문제:**
- 현재 `simple_line_evaluation()` 함수는 **끝점 좌표 기반**
- AirLine의 끝점 검출이 부실함
- 각도와 거리 유사도 계산이 끝점에 의존

**현재 코드:**
```python
# optimization.py:90-98
gt_angle = np.arctan2(gt_y2 - gt_y1, gt_x2 - gt_x1)
det_angle = np.arctan2(det_y2 - det_y1, det_x2 - det_x1)
angle_diff = abs(gt_angle - det_angle)
angle_similarity = 1.0 - (angle_diff / np.pi)
```

**개선 방향:**
- **직선 방정식 기반 평가:**
  - 기울기(slope): `m = (y2-y1)/(x2-x1)`
  - 절편(intercept): `c = y1 - m*x1`
  - 직선: `y = mx + c` 또는 `Ax + By + C = 0`

```python
def line_equation_evaluation(detected_line, gt_line):
    """
    직선 방정식 기반 평가

    Args:
        detected_line: [x1, y1, x2, y2]
        gt_line: [x1, y1, x2, y2]

    Returns:
        score: float [0, 1]
    """
    # 1. 직선 방정식으로 변환
    # Ax + By + C = 0 형태
    def line_to_equation(x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        norm = np.sqrt(A**2 + B**2)
        return A/norm, B/norm, C/norm

    A1, B1, C1 = line_to_equation(*gt_line)
    A2, B2, C2 = line_to_equation(*detected_line)

    # 2. 방향 유사도 (법선 벡터 내적)
    direction_sim = abs(A1*A2 + B1*B2)

    # 3. 거리 유사도 (점-직선 거리)
    # GT 직선에서 검출 직선까지 평균 거리
    dist1 = abs(A1*x2 + B1*y2 + C1)  # 검출 점1
    dist2 = abs(A1*x2' + B1*y2' + C1)  # 검출 점2
    avg_dist = (dist1 + dist2) / 2

    distance_sim = 1.0 - (avg_dist / threshold)

    return direction_sim * 0.7 + distance_sim * 0.3
```

---

### 5. 환경 표현 개선 필요 🔧

**현재 환경 벡터 (6D):**
```python
# environment_independent.py:85-93
env = {
    'brightness': float(brightness_score),      # 밝기
    'contrast': float(contrast_score),          # 대비
    'edge_density': float(edge_score),          # 엣지 밀도
    'texture_complexity': float(texture_score), # 텍스처 복잡도
    'blur_level': float(blur_score),            # 블러
    'noise_level': float(noise_score)           # 노이즈
}
```

**문제:**
- 이미지의 **의미적 특성**을 충분히 반영 못함
- 그림자 여부, 용접 비드 노이즈 등 도메인 특화 특징 없음

**개선 방안:**

#### 1) CLIP 기반 환경 특징 추가
```python
import torch
import clip

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 텍스트 프롬프트
prompts = [
    "welding line with heavy shadow",
    "clear welding line without shadow",
    "noisy welding surface with beads",
    "clean welding surface"
]

def extract_clip_features(image):
    image_input = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        # 코사인 유사도
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    return {
        'has_heavy_shadow': float(similarity[0, 0]),
        'has_clear_view': float(similarity[0, 1]),
        'has_noise': float(similarity[0, 2]),
        'is_clean': float(similarity[0, 3])
    }
```

#### 2) PSNR/SSIM 추가
```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def add_image_quality_metrics(image, reference=None):
    """
    이미지 품질 메트릭 추가

    Args:
        image: 현재 이미지
        reference: 참조 이미지 (None이면 전처리 전/후 비교)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur를 참조로 사용
    if reference is None:
        reference = cv2.GaussianBlur(gray, (5, 5), 0)

    # PSNR (높을수록 좋음)
    psnr = peak_signal_noise_ratio(gray, reference)
    psnr_normalized = 1.0 - np.clip(psnr / 50.0, 0, 1)  # 낮으면 어려움

    # SSIM (1에 가까울수록 좋음)
    ssim_val = structural_similarity(gray, reference)
    ssim_score = 1.0 - ssim_val  # 낮으면 어려움

    return {
        'psnr_difficulty': float(psnr_normalized),
        'ssim_difficulty': float(ssim_score)
    }
```

**최종 환경 벡터:** 6D → 12D
- 기존 6D (brightness, contrast, edge_density, texture, blur, noise)
- CLIP 4D (shadow, clear, noisy, clean)
- Quality 2D (PSNR, SSIM)

---

### 6. 워크스테이션 호환성 문제 🔴

**문제:** `CRG311.pyd` (Windows 전용) → Linux에서 작동 안함

**의존성 확인:**
```bash
✓ Python: 3.11.14
✓ torch: 2.6.0+cu124
✓ opencv: 4.12.0
✓ botorch: 0.16.0
✓ ultralytics: 8.3.227
✗ CRG311: ModuleNotFoundError
```

**해결 중:** AirLine 공식 리포에서 Linux 빌드 설치

---

## 🎯 현재 작업 (2025.11.11 19:20)

### 우선순위: 결정변수 업데이트 + 평가 메트릭 수정

#### Step 1: 평가 메트릭 변경 [진행중]
- [x] AirLine Linux 설치 완료
- [ ] `line_equation_evaluation()` 함수 구현
- [ ] `simple_line_evaluation()` 대체
- [ ] 테스트 실행

#### Step 2: RANSAC 가중치 추가 [진행중]
- [ ] BOUNDS를 6D → 9D로 확장
- [ ] `objective_function()`에서 3개 파라미터 추가
- [ ] full_pipeline.py 연결 확인
- [ ] 테스트 실행

#### Step 3: 로그 정리 [진행중]
- [ ] 불필요한 print문 제거
- [ ] 진행상황만 간결하게 표시
- [ ] iteration별 로그 파일 저장
- [ ] 검증 로직 확인 (각 step당 1회만)

#### Step 4: 실험 실행
- [ ] 수정된 코드로 빠른 테스트 (3-5 iterations)
- [ ] 결과 확인 (CVaR > 0.01 기대)
- [ ] 성공 시: 긴 실험 (20-30 iterations)
- [ ] 시각화 생성

### 이후 작업

#### Step 5: 환경 특징 강화
- [ ] CLIP 설치 및 테스트
- [ ] `extract_clip_features()` 구현
- [ ] PSNR/SSIM 메트릭 추가
- [ ] `environment_independent.py` 업데이트

#### Step 6: 환경 변수 통합
- [ ] 이미지별 환경 벡터 추출 및 저장
- [ ] BOUNDS를 9D → 15D로 확장
- [ ] GP 입력을 (x, z)로 변경
- [ ] 테스트 실행

#### Step 7: 판타지 관측 구현
- [ ] BoRisk 논문 알고리즘 재확인
- [ ] CVaR Knowledge Gradient 획득함수 구현
- [ ] Fantasy observation 구현
- [ ] 전체 파이프라인 테스트

---

## 📝 작업 로그

### 2025.11.11 - 초기 분석

**완료:**
- ✅ 코드 워크스테이션 이식 완료
- ✅ Python 환경 구성 (conda env: weld2024_mk2)
- ✅ 의존성 설치: torch, opencv, botorch, ultralytics
- ✅ 데이터셋 경로 확인: `../dataset/`
- ✅ 6가지 주요 문제점 발견 및 분석

**발견한 파일들:**
```
/home/jeongho/projects/graduate/
├── BO_optimization/          # 현재 작업 디렉토리
│   ├── optimization.py
│   ├── full_pipeline.py
│   ├── environment_independent.py
│   ├── models/best.pt
│   └── [기타 .py 파일들]
├── dataset/                  # 데이터셋
│   ├── images/test/         # 119장 이미지
│   └── ground_truth.json
└── YOLO_AirLine/            # AirLine 관련
    ├── AirLine_assemble_test.py
    ├── CRG311.pyd           # Windows 전용 (문제)
    └── CRG/extractC/CRGandLP.cpp  # C++ 소스
```

**다음:** 평가 메트릭 + RANSAC 가중치 수정 후 실험

---

## 📝 작업 로그 (계속)

### 2025.11.11 19:15 - Git Push 완료

**완료:**
- ✅ Git push 성공 (3개 커밋)
- ✅ 시각화 모듈 생성 (4종 그래프)
- ✅ 빠른 테스트 실행 (CVaR=0.0011, 낮음)

**문제 발견:**
- CVaR 값이 매우 낮음 (0.0011) → 라인 검출 실패
- 끝점 기반 평가의 한계 확인

**결정:**
- 평가 메트릭을 직선 방정식 기반으로 변경
- RANSAC 가중치를 결정변수에 추가 (6D → 9D)
- 로그 정리 (토큰 낭비 방지)

---

---

### 2025.11.11 19:28 - 평가 메트릭 + RANSAC 완료

**완료:**
- ✅ 평가 메트릭 변경: 끝점 → 직선 방정식 기반
  - `line_equation_evaluation()` 함수 추가
  - Ax + By + C = 0 형식으로 직선 표현
  - 방향 유사도 (법선 벡터 내적) + 평행 거리 계산
  - 가중치: direction 60%, distance 40%

- ✅ RANSAC 가중치 추가: 6D → 9D 확장
  - BOUNDS 업데이트: [6D] → [9D]
  - Sobol 엔진 차원 수정: dimension=6 → 9
  - objective_function에 RANSAC 파라미터 전달
  - 결과 저장 시 9D 모두 포함

- ✅ 로깅 최적화
  - 초기화 단계: 단일 라인으로 축약
  - 반복 단계: 핵심 정보만 출력
  - 상세 로그를 logs/iter_XXX.json 파일로 저장
  - 토큰 낭비 최소화

- ✅ 평가 로직 검증
  - objective_function은 각 스텝당 정확히 1회 호출
  - CVaR 계산은 직접 평가 사용 (GP 예측 아님)
  - 113개 이미지 전체 평가 후 worst 30% 평균

**수정된 파일:**
- `optimization.py`:
  - `line_equation_evaluation()` 함수 추가 (39-116라인)
  - BOUNDS 9D 확장 (33-36라인)
  - Sobol 9D 적용 (296라인)
  - objective_function에 RANSAC 추가 (238-240라인)
  - 로깅 정리 (305, 328-332, 371-423라인)
  - 반복별 로그 파일 저장 (378-400라인)
  - 결과 저장 9D (539-541라인)
  - 결과 출력 9D (513-515라인)

**현재 상태:**
- 🔄 테스트 실행 중 (Background ID: f28d36)
- 명령: 3 init + 2 iter, α=0.3
- 로그: new_test.log

**다음:**
- 테스트 결과 확인
- CVaR 개선 확인 (이전 0.0011 대비)
- 성공 시 full experiment (20-30 iterations)

**마지막 업데이트:** 2025.11.12 23:50

---

## 🔥 최신 상황 (2025.11.12)

### 환경 변경: Linux → Windows 회귀
- **이유**: CRG311 Linux 빌드 segfault 문제 해결 불가
- **현재 환경**: Windows 10, Python 3.12.0, conda weld2024_mk2
- **경로**: `C:\Users\user\Desktop\study\task\graduate\graduate_master\BO_optimization`

### ✅ 완료된 작업 (11.12)

#### 1. BoRisk KG 구현 완료 ✨
- **파일**: `borisk_kg.py` (Opus와 Linux Claude Code로 작성)
- **내용**: Simplified CVaR-KG 획득 함수 구현
- **통합**: `optimization.py`에서 `optimize_borisk()` 사용
- **상태**: ✅ 작동 확인 완료!

#### 2. 실험 완료 (alpha=0.3)
- **결과 파일**: `results/bo_cvar_20251112_233205.json`
- **설정**: 113 images, n_w=15, 3 iterations
- **성능**:
  - Initial CVaR: **0.5813**
  - Peak CVaR: **0.8230** (iteration 5)
  - Final CVaR: **0.6410**
  - **Improvement: +10.27%** 🎉

#### 3. 환경 벡터 시스템 완료
- **파일**: `environment_independent.py`
- **차원**: 6D (brightness, contrast, edge_density, texture, blur, noise)
- **통합**: GP 입력 15D = params 9D + env 6D
- **상태**: ✅ 작동 중

### ❌ 현재 문제점

#### 1. RANSAC 단일 라인 버그 (재발!)
**위치**: `full_pipeline.py:234`
**함수**: `weighted_ransac_line()`

```python
# Line 234: RANSAC iteration
i1, i2 = rng.choice(len(all_lines), size=2, replace=False, p=probs)
# ↑ 라인이 1개만 있으면 크래시!
```

**증상**:
- ERROR: `Cannot take a larger sample than population when replace is False`
- 특정 이미지 (idx=1 등)에서만 발생
- 실험 초기화는 성공, iteration 중 크래시

**이전 수정**: Line 176-195에 early return 로직 추가했었음
**문제**: 왜 여전히 line 234에 도달하는가? 🤔

**해결 방안**:
```python
# Line 233-234 수정 필요
if len(all_lines) < 2:
    # 라인 1개일 때 처리 (이미 176-195에 있지만 재확인)
    return single_line_handling()

# 라인 2개 이상일 때만 RANSAC
i1, i2 = rng.choice(len(all_lines), size=2, replace=False, p=probs)
```

#### 2. Alpha=0.1 실험 실패
- **시도**: 2번 (bash ID: e75363, 2e5739)
- **상태**: 둘 다 크래시 (exit code 127)
- **원인**: RANSAC 버그 + Windows shell 문제?
- **대책**: 버그 수정 후 재실행 필요

### 📊 실험 이력

| 날짜 | Alpha | Images | Iterations | CVaR (초기→최종) | 개선율 | 상태 |
|------|-------|---------|-----------|----------------|--------|------|
| 11.12 23:32 | 0.3 | 113 | 3 | 0.581→0.641 | **+10.3%** | ✅ 완료 |
| 11.12 23:27 | 0.3 | 10 | 3 | 0.815→0.815 | -0.04% | ✅ 완료 |
| 11.12 23:19 | 0.3 | ? | ? | ? | ? | ✅ 완료 |
| 11.12 23:45+ | 0.1 | 113 | 15 | - | - | ❌ 크래시 |
| 11.12 23:45+ | 0.1 | 113 | 15 | - | - | ❌ 크래시 |

### 🎯 다음 할 일 (우선순위)

1. **RANSAC 버그 완전 수정** (최우선!)
   - [ ] Line 176-195 로직 재확인
   - [ ] Line 233에 추가 방어 로직
   - [ ] 테스트: 라인 0개, 1개, 2개 케이스

2. **새 실험 실행**
   - [ ] Alpha=0.1 (worst 10%, 극단 케이스 초점)
   - [ ] Alpha=0.2 (비교용)
   - [ ] Alpha=0.3 재실행 (더 많은 iterations)

3. **시각화 생성**
   - [ ] CVaR 개선 그래프
   - [ ] Alpha별 비교
   - [ ] 환경별 성능 분포

4. **MD 문서 업데이트**
   - [x] TRACKING.md 업데이트
   - [ ] NEXT_SESSION.md 업데이트
   - [ ] Claude.md 동기화

**마지막 업데이트:** 2025.11.12 23:50
