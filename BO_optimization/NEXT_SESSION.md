# 🚨 긴급 세션 가이드 - 2025-11-13

**상황**: 오늘까지 실험 결과를 내지 못하면 졸업 불가
**환경**: Windows 로컬 (리눅스 segfault로 회귀, 코드 복붙 사용 중)
**현재 상태**: Dimension mismatch 버그 수정 완료, 자동 라벨링 스크립트 생성 완료

---

## 📅 작업 계획 (2025.11.13 ~ 2025.11.14)

### 🎯 **오늘 (2025.11.13) 목표**

1. ✅ **CVaR 계산 방식 수정** - BoRisk 논문 방식으로 (GP posterior 샘플링)
2. ✅ **판타지 관측 및 획득 함수 동작 검증**
3. ✅ **자동 라벨링 스크립트 완성** - AirLine_assemble_test.py 6개 점 활용
4. ✅ **백그라운드 실험 결과 확인**
5. ✅ **환경 벡터 개선** - 일관성 확보 (선 검출 실패 이미지 분석)

### 🎯 **내일 (2025.11.14) 목표**

1. ✅ **CLIP 적용** - Zero-shot / fine-tuning으로 직관적 환경 인식
2. ✅ **학회/저널 준비** - BoRisk 수정 가미

---

## ✅ 완료된 작업 (2025.11.13 세션)

### 1. 파이프라인 아키텍처 문서화 완료 ✓
- **PIPELINE_SUMMARY.md** 생성: 전체 파이프라인 구조 명확화
- AirLine 원본 vs 커스텀 RANSAC 비교 분석
- BO 최적화를 위한 커스텀 RANSAC 구현 이유 설명
- 파일 위치: `BO_optimization/PIPELINE_SUMMARY.md`

### 2. RANSAC Single-Line Bug 완전 수정 ✓
- **문제**: `weighted_ransac_line()`에서 1개 라인만 있을 때 크래시
  - Line 318: `rng.choice(len(all_lines), size=2, replace=False, p=probs)`
  - Error: "Cannot take a larger sample than population when replace is False"
- **수정**: `full_pipeline.py:316-318`에 방어 로직 추가
  ```python
  # ✅ 방어 로직: RANSAC은 최소 2개 라인 필요
  if len(all_lines) < 2:
      return None
  ```
- **검증**: 테스트 완료 - 6개 이미지 평가에서 RANSAC 에러 0건 ✅
- **이전 실패 케이스**: alpha=0.1 실험 2건 모두 이 버그로 실패
- **현재 상태**: RANSAC 버그 완전 해결!

### 3. Dimension Mismatch 버그 수정 완료 ✓
- **문제**: borisk_kg.py에서 하드코딩된 `9`가 bounds 8D와 불일치
  - Line 275: `torch.rand(n_candidates, 9, ...)` ← 하드코딩
  - Line 289: `torch.rand(1, 9, ...)` ← 하드코딩
- **수정**: `bounds.shape[1]`로 동적 처리 (Line 275, 290)
- **테스트**: 5개 이미지로 실험 성공 ✅
- **결과**: Dimension 에러 없음, 실험 정상 실행됨

### 4. 자동 라벨링 스크립트 작성 완료 ✓
- **파일**: `auto_labeling.py`
- **기능**: AirLine_assemble_test.py 사용해 6개 점 자동 추출
- **출력**: ground_truth_auto.json (GT 포맷과 동일)
- **사용법**: `python auto_labeling.py --image_dir ../dataset/images/test --output ../dataset/ground_truth_auto.json`
- **현재 상태**: Upper 점 계산 로직 임시 구현됨 → **수정 필요**: AirLine_assemble_test는 6개 점을 모두 제공하므로 직접 사용

### 5. Git Push 완료 ✓
- **커밋**: `4b0d73a` - "FIX: Dimension mismatch 버그 수정 + 자동 라벨링 스크립트 추가"
- **변경 파일**: borisk_kg.py, auto_labeling.py, NEXT_SESSION.md

---

## 🎯 현재 세션 작업 (우선순위)

### 🚨 Priority 0: CVaR 계산 방식 수정 (최우선!)

**현재 문제**:
현재 코드는 **실제 이미지를 평가**해서 CVaR을 계산하고 있음 (`evaluate_on_w_set` → `detect_with_full_pipeline`)

**BoRisk 논문 방식**:
- 초기 평가로 GP 모델 학습
- 이후 **GP posterior 샘플링**으로 CVaR 추정
- 실제 평가는 최종 후보만 진행
- **핵심**: 판타지 관측을 통해 미래 성능 예측

**참고 자료**:
- https://github.com/saitcakmak/BoRisk
- BoRisk 논문: "Bayesian Optimization under Risk" (2020)
- BoTorch qMultiFidelityKnowledgeGradient 구현

**수정 필요 부분**:
1. `optimization.py`의 `evaluate_on_w_set()` 함수
   - 현재: 실제 이미지 평가
   - 변경: GP posterior 샘플링
2. GP posterior에서 샘플링하는 함수 추가
3. CVaR을 GP로 추정하도록 수정
4. 판타지 관측 구조 검증

**작업 단계**:
```python
# 1. GP posterior 샘플링 함수 추가
def sample_from_gp_posterior(gp, x, w_set, n_samples=100):
    """GP posterior에서 샘플링하여 CVaR 추정"""
    # (x, w) 쌍 생성
    # GP.posterior() 사용
    # rsample()로 샘플링
    # CVaR 계산

# 2. evaluate_on_w_set 수정
# 실제 평가 대신 GP 샘플링 사용 (초기 평가 후)
```

---

### 🎯 Priority 1: 판타지 관측 및 획득 함수 동작 검증

**목표**: 현재 SimplifiedBoRiskKG가 제대로 작동하는지 확인

**검증 항목**:
1. ✅ **판타지 샘플 생성**: `posterior.rsample()` 동작 확인
2. ✅ **CVaR 계산**: worst α% 선택이 올바른가
3. ✅ **개선도 계산**: `max(0, fantasy_cvar - current_best_cvar)` 로직 확인
4. ✅ **획득 함수 최적화**: 후보 선택이 합리적인가

**테스트 방법**:
```bash
# 작은 데이터셋으로 디버깅
python optimization.py --iterations 3 --n_initial 2 --alpha 0.3 --max_images 5 --n_w 3 --debug
```

**로깅 추가**:
```python
# borisk_kg.py에 로깅 추가
print(f"Fantasy CVaR: {fantasy_cvar:.4f}, Current: {self.current_best_cvar:.4f}, Improvement: {improvement:.4f}")
```

---

### 🎯 Priority 2: 자동 라벨링 스크립트 수정

**현재 문제**:
- `auto_labeling.py`에서 Upper 점 계산이 임시 구현됨
- AirLine_assemble_test.py는 6개 점을 모두 제공함 (longi 4개 + collar 2개)

**수정 방향**:
1. AirLine_assemble_test.py의 `run_airline_test()` 함수 직접 사용
2. 반환되는 6개 점을 그대로 사용
3. 특정할 수 없는 경우 휴리스틱 방법 사용 (사용자가 수정 예정)

**구현**:
```python
# auto_labeling.py 수정
from AirLine_assemble_test import run_airline_test

def auto_label_image(image_path):
    # AirLine_assemble_test 실행
    result = run_airline_test(image_path)

    if result and len(result) == 6:
        # 6개 점 모두 반환됨
        longi_left_lower, longi_right_lower, longi_left_upper, longi_right_upper, collar_left_lower, collar_left_upper = result
        return format_coordinates(result)
    else:
        # 휴리스틱 방법 (사용자 수정 예정)
        return None
```

**테스트**:
```bash
# 소량 테스트
python auto_labeling.py --image_dir ../dataset/images/test --output test_auto_gt.json --max_images 10

# 결과 확인
cat test_auto_gt.json | head -30
```

---

### 🎯 Priority 3: 백그라운드 실험 결과 확인

**현재 실행 중인 실험들**:
- 9개 background bash 프로세스
- 테스트 실험들 (max_images=3, 5)
- alpha=0.1 실험 (iterations=15)

**작업**:
1. 각 프로세스 상태 확인
2. 완료된 실험 결과 분석
3. 실패한 실험 원인 파악

**명령어**:
```bash
# 프로세스 확인
BashOutput tool로 각 bash_id 확인

# 결과 파일 확인
ls -lt results/ | head -10
cat results/bo_cvar_*.json | tail -1
```

---

### 🎯 Priority 4: 환경 벡터 개선 - 일관성 확보

**목표**: 선 검출이 잘 안되는 이미지들에서 일관적인 환경 파라미터 생성

**현재 문제**:
- CVaR 값이 일관성이 없음
- 도출된 파라미터를 적용한 결과가 좋지 못한 경우가 많음
- 실패 케이스를 제대로 구분하지 못함

**작업 단계**:

#### 1. 실패 이미지 분석
```python
# failure_analysis.py 작성
def analyze_failed_images(results_json):
    """
    실패 케이스 분석
    - score < 0.5인 이미지 추출
    - 환경 파라미터 확인
    - 공통 패턴 찾기
    """
    failed_images = [img for img in results if img['score'] < 0.5]

    # 환경 파라미터 분포 확인
    env_params = [extract_environment(img_path) for img_path in failed_images]

    # 클러스터링
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=3).fit(env_params)

    return failed_images, env_params, clusters
```

#### 2. 환경 벡터 개선
**현재 6D 환경 벡터**:
- brightness, contrast, edge_density, texture_complexity, blur_level, noise_level

**추가 고려사항**:
- 선 검출에 큰 장애가 되는 요소 파악
- 실패 케이스에서 일관된 환경 파라미터 도출

**개선 방향**:
```python
# environment_improved.py
def extract_environment_improved(image):
    """
    개선된 환경 벡터 추출
    - 기존 6D 유지
    - 추가: 선 검출 난이도 관련 특징
      - 용접선 명확도
      - 배경 복잡도
      - 조명 균일도
    """
    base_env = extract_environment(image)  # 기존 6D

    # 추가 특징
    weld_line_clarity = compute_weld_line_clarity(image)
    background_complexity = compute_background_complexity(image)
    lighting_uniformity = compute_lighting_uniformity(image)

    return np.concatenate([base_env, [weld_line_clarity, background_complexity, lighting_uniformity]])
```

#### 3. 일관성 테스트
```bash
# 실패 이미지들에 대해 환경 파라미터 추출
python failure_analysis.py --results results/bo_cvar_*.json --output failure_report.json

# 클러스터링 결과 시각화
python visualize_clusters.py --input failure_report.json
```

---

## 📅 내일 작업 계획 (2025.11.14)

### 🎯 Priority 1: CLIP 적용

**목표**: Zero-shot 또는 fine-tuning으로 직관적인 환경 인식

**작업 단계**:

#### 1. CLIP 모델 설치
```bash
pip install transformers torch torchvision
pip install openai-clip
```

#### 2. Zero-shot 환경 분류
```python
# clip_environment.py
import torch
import clip
from PIL import Image

def classify_environment_with_clip(image_path):
    """
    CLIP을 사용한 환경 분류
    """
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    # 환경 카테고리 정의
    categories = [
        "clear welding line with good contrast",
        "blurry welding line with low contrast",
        "welding line with bright reflection",
        "welding line with complex background",
        "dark welding line with shadows"
    ]

    # 이미지 로드
    image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")
    text = clip.tokenize(categories).to("cuda")

    # 유사도 계산
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity[0].cpu().numpy()
```

#### 3. Fine-tuning (선택사항)
```python
# fine_tune_clip.py
# 실패/성공 케이스로 CLIP fine-tuning
# 목표: 선 검출 난이도 직접 예측
```

---

### 🎯 Priority 2: 학회/저널 준비

**목표**: BoRisk에 수정을 가미하여 학회/저널 제출 준비

**작업 항목**:
1. **알고리즘 개선**
   - BoRisk + 환경 인식 (CLIP)
   - 제안: "Environment-Aware Risk-Averse Bayesian Optimization"

2. **실험 결과 정리**
   - 다양한 alpha 실험 결과
   - CVaR 개선도 분석
   - 실패 케이스 분석

3. **Figure 생성**
   - 최적화 과정 시각화
   - alpha별 성능 비교
   - 환경별 성능 분포

4. **논문 초안 작성**
   - Abstract
   - Introduction
   - Method
   - Experiments
   - Conclusion

---

## 🐛 기술적 이슈

### 1. 환경 문제

#### Linux Workstation (실패):
- Segmentation fault 발생
- 원인: CRG311 라이브러리 의존성 문제
- 상태: 포기, Windows로 회귀

#### Windows Local (현재):
- 실행 가능
- 코드 복붙 사용 중 (깔끔하지 않음)
- Git 브랜치 분리 필요

### 2. Git 관리

**현재 상황**:
- Linux 수정사항: 경로 문제 해결
- Windows 환경: 별도 코드 복붙
- 걱정: 브랜치 분리하면 경로 충돌 가능

---

## 🚀 빠른 시작 명령어

### 환경 설정
```bash
# conda 환경 활성화
conda activate weld2024_mk2

# 작업 디렉토리
cd C:/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
```

### 1. CVaR 계산 방식 테스트
```bash
# GP posterior 샘플링 테스트
python test_gp_sampling.py --max_images 5
```

### 2. 자동 라벨링
```bash
# 소량 테스트
python auto_labeling.py --image_dir ../dataset/images/test --output test_auto_gt.json --max_images 10

# 전체 실행
python auto_labeling.py --image_dir ../dataset/images/test --output ../dataset/ground_truth_auto.json
```

### 3. 실패 이미지 분석
```bash
# 실패 케이스 분석
python failure_analysis.py --results results/bo_cvar_*.json --output failure_report.json

# 시각화
python visualize_clusters.py --input failure_report.json
```

### 4. 전체 실험
```bash
# 최종 실험
python optimization.py --iterations 20 --n_initial 10 --alpha 0.3
```

---

## 📊 성공 기준

### 오늘 (2025.11.13) 달성 목표:
1. ✅ CVaR 계산 방식 수정 완료
2. ✅ 판타지 관측 동작 검증 완료
3. ✅ 자동 라벨링 스크립트 완성
4. ✅ 백그라운드 실험 결과 확인
5. ✅ 환경 벡터 개선 (실패 이미지 일관성 확보)

### 내일 (2025.11.14) 달성 목표:
1. ✅ CLIP 적용 및 Zero-shot 환경 분류
2. ✅ 학회/저널 논문 초안 작성
3. ✅ 실험 결과 정리 및 Figure 생성

---

## 💡 중요 메모

### BoRisk 핵심 이해
**현재 구현의 문제**:
- 매 iteration마다 실제 이미지 평가 (느림, 비효율적)
- w_set의 모든 이미지를 평가 (113개 전체 평가)

**BoRisk 올바른 구현**:
- 초기 평가로 GP 학습
- 이후 **GP posterior 샘플링**으로 CVaR 추정
- 실제 평가는 선택된 후보만 진행 (매 iteration 1개)
- 속도: 현재의 1/10로 단축 가능

### AirLine_assemble_test.py 활용
- 6개 점을 모두 제공하므로 Upper 점 계산 로직 불필요
- 직접 사용 가능
- 특정 불가 케이스는 휴리스틱 (사용자 수정 예정)

---

**마지막 업데이트**: 2025-11-13 01:55
**다음 작업**: CVaR 계산 방식 수정 + 자동 라벨링 + 환경 벡터 개선
**Status**: ✅ Dimension mismatch 수정 완료, 작업 시작 준비됨

**화이팅! 졸업하자! 🎓**
