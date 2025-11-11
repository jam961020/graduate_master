# Claude Development Guide
## BoRisk CVaR Optimization for Welding Line Detection

Repository: https://github.com/jam961020/graduate_master

**최종 업데이트: 2025.11.11 19:35**

## 📌 대전제

- **이 프로젝트의 핵심은 BoRisk 알고리즘의 올바른 구현**
- BoRisk = Risk-aware Bayesian Optimization using CVaR (Conditional Value at Risk)
- 논문: ["Bayesian Optimization under Risk" (BoRisk)](https://arxiv.org/abs/2011.05939)
- **⚠️ 하드코딩으로 우회하지 말고 문제의 본질을 해결하라**
- **⚠️ 임시 해결책 사용 시 반드시 TODO 주석을 남겨라**

---

## 🤖 Claude 협업 환경

### Claude 성능 비교
- **Claude Chat (Opus 4.1)**: 복잡한 문제 해결, 전체 구조 설계, 디버깅에 강함
- **Claude Code**: 빠른 코드 수정, 반복 작업, 로컬 파일 직접 편집에 유리
- **추천**: 설계/디버깅은 Chat, 구현/수정은 Code 사용

---

## 📁 프로젝트 구조

```
graduate_master/
├── optimization.py           # BoRisk CVaR 최적화 메인
├── full_pipeline.py         # YOLO + AirLine 통합 파이프라인
├── AirLine_assemble_test.py # AirLine 알고리즘 구현
├── yolo_detector.py         # YOLO 검출기 래퍼
├── evaluation.py            # 평가 메트릭
├── environment.py           # 환경 특징 추출
├── dataset/
│   ├── images/test/        # 119장 용접 이미지
│   └── ground_truth.json   # GT 라벨
├── models/
│   └── best.pt             # YOLO 모델
├── results/                # 실험 결과 JSON
├── logs/                   # 실행 로그
├── PROJECT_GUIDELINES.md   # 프로젝트 지침서
└── Claude.md              # 이 파일
```

---

## 🎯 현재 작업 상태 (2025.11.11 19:35)

### ✅ 완료된 작업

#### 1. CRG311 Linux 빌드 (완료 19:00)
- AirLine 공식 리포에서 C++ 소스 컴파일
- pybind11로 Linux .so 생성
- 경로 수정 및 lazy initialization 적용

#### 2. 평가 메트릭 변경 (완료 19:28)
- **끝점 기반 → 직선 방정식 기반**
- `line_equation_evaluation()` 함수 추가 (optimization.py:39-116)
- Ax + By + C = 0 형식으로 정규화
- 방향 유사도 (법선 벡터 내적) + 평행 거리
- 가중치: direction 60%, distance 40%

#### 3. RANSAC 가중치 최적화 (완료 19:28)
- **6D → 9D 확장**
- BOUNDS 업데이트: 9D [AirLine 6D + RANSAC 3D]
- `ransac_center_w`, `ransac_length_w`, `ransac_consensus_w` 추가
- Sobol 엔진 차원 수정: dimension=9
- objective_function에 파라미터 전달 구현

#### 4. 로깅 최적화 (완료 19:28)
- 화면 출력 최소화 (토큰 절약)
- 상세 로그를 `logs/iter_XXX.json`로 파일 저장
- 각 반복마다 9D 파라미터, CVaR, 획득함수 값 기록

### 🔄 진행 중

- **테스트 실행 중** (백그라운드 프로세스)
- 명령: `python optimization.py --iterations 2 --n_initial 3 --alpha 0.3`
- 로그: `new_test.log`, `logs/iter_*.json`

### 🔴 남은 주요 문제점

#### 1. CVaR 계산 방식 (Critical)
- **현재**: 직접 평가 사용 (모든 이미지에 대해 실제로 실행)
- **문제**: BoRisk 논문에서는 GP를 활용한 CVaR 계산 필요
- **필요한 것**:
  - GP로부터 예측 분포 샘플링
  - 샘플링된 분포에서 CVaR 계산
  - TODO: `optimization.py:217-273` 수정 필요

#### 2. 환경 변수 미사용 (Critical)
- `environment_independent.py`에 6D 환경 벡터 구현되어 있으나 **optimization.py에서 전혀 사용 안 함**
- GP가 (x, z) → y 학습하지 않고 x → y만 학습 (일반 BO와 동일)
- BoRisk의 핵심인 이미지별 환경 컨디셔닝 누락
- TODO: 9D → 15D 확장 (params 9D + env 6D)
- `optimization.py`의 BOUNDS가 6D만 정의 (9D로 확장 필요)
- ransac_center_w, ransac_length_w, ransac_consensus_w 하드코딩됨

#### 3. 판타지 관측 미구현 (Critical)
- **BoRisk의 핵심 알고리즘 완전히 누락**
- 현재는 단순 Vanilla BO + CVaR 목적함수
- 필요: CVaR Knowledge Gradient 획득함수, fantasy observation

#### 4. 평가 메트릭 문제 (High)
- 현재: 끝점 좌표 기반 평가
- 문제: AirLine의 끝점 검출이 부실함
- 해결: 직선 방정식 기반 (기울기 + 절편) 평가로 변경 필요

#### 5. 환경 표현 개선 필요 (Medium)
- 현재 6D 환경 벡터가 이미지 특성 충분히 반영 못함
- 추가 필요: CLIP 기반 그림자/노이즈 검출, PSNR/SSIM 메트릭

#### 6. 워크스테이션 호환성 (Blocker)
- `CRG311.pyd` (Windows 전용) → Linux 환경에서 import 실패
- AirLine 코어 모듈 `crg.desGrow()` 사용 불가
- 해결: github.com/sair-lab/AirLine의 Linux 빌드 설치 필요

### 완료된 작업
- ✅ 코드 워크스테이션 이식
- ✅ Python 환경 구성 (torch, opencv, botorch, ultralytics)
- ✅ 데이터셋 경로 확인 (../dataset/)
- ✅ 문제점 분석 완료

### 진행중 작업
- 🔄 AirLine 공식 리포지토리에서 Linux 빌드 설치
- 🔄 평가 메트릭을 직선 방정식 기반으로 변경
- 🔄 환경 특징에 CLIP, PSNR/SSIM 추가

### 예정 작업
- 📋 RANSAC 가중치를 최적화 파라미터에 추가 (6D → 9D)
- 📋 환경 변수를 GP에 통합 (9D → 15D: params + env)
- 📋 판타지 관측 구현 (BoRisk 알고리즘)
- 📋 실험 실행 및 결과 분석

---

## 🚀 빠른 실행 명령어

### 워크스테이션 환경
- 경로: `/home/jeongho/projects/graduate/BO_optimization`
- Python: 3.11.14 (weld2024_mk2 환경)
- GPU: CUDA 12.4 available

```bash
# 데이터셋 경로 (상위 디렉토리)
# ../dataset/images/test/  (119장)
# ../dataset/ground_truth.json

# 기본 테스트 (5회)
python optimization.py --iterations 5 --n_initial 10 --alpha 0.3

# 표준 실행 (20회)
python optimization.py --iterations 20 --n_initial 15 --alpha 0.3

# 전체 실행 (30회)
python optimization.py --iterations 30 --n_initial 20 --alpha 0.2
```

---

## 🔧 주요 파라미터

### AirLine 파라미터 (6D)
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| edgeThresh1 | [-23.0, 7.0] | -3.0 | Q 프리셋 엣지 임계값 |
| simThresh1 | [0.5, 0.99] | 0.98 | Q 프리셋 유사도 |
| pixelRatio1 | [0.01, 0.15] | 0.05 | Q 프리셋 픽셀 비율 |
| edgeThresh2 | [-23.0, 7.0] | 1.0 | QG 프리셋 엣지 임계값 |
| simThresh2 | [0.5, 0.99] | 0.75 | QG 프리셋 유사도 |
| pixelRatio2 | [0.01, 0.15] | 0.05 | QG 프리셋 픽셀 비율 |

### 환경 벡터 (4D)
- brightness: [0, 1] - 평균 밝기
- contrast: [0, 1] - 표준편차/128
- edge_density: [0, 1] - Canny 엣지 비율
- texture: [0, 1] - Laplacian 분산

---

## 🐛 자주 발생하는 문제

### 1. CRG311 import 실패 (Linux)
```bash
# 증상: ModuleNotFoundError: No module named 'CRG311'
# 원인: CRG311.pyd는 Windows 전용
# 해결: AirLine 공식 리포에서 Linux 빌드 설치
git clone https://github.com/sair-lab/AirLine.git
cd AirLine
# 설치 방법은 리포의 README 참조
```

### 2. NumPy 버전 충돌
```bash
# 증상: sklearn import 시 NumPy 2.x 에러
# 해결: NumPy 1.x로 다운그레이드
pip install "numpy>=1.23,<2.0" --force-reinstall
```

### 3. AirLine 로깅 과다
```python
# 해결: monkey patching
full_pipeline.detect_lines_in_roi = quiet_detect_lines_in_roi
```

### 4. GP 학습 실패
```python
# 해결: Y 정규화
Y_normalized = (Y - Y.mean()) / (Y.std() + 1e-6)
```

### 5. 획득 함수 0 반환
```python
# 해결: 초기 샘플 증가, 탐험 파라미터 조정
n_initial = 20  # 15 → 20
beta = 2.0      # UCB 탐험 증가
```

---

## 📊 성능 지표

### 현재 최고 성능
- CVaR (α=0.3): 0.812
- 개선율: +8.3%
- 최적 파라미터:
  ```
  edgeThresh1: -5.23
  simThresh1: 0.923
  pixelRatio1: 0.082
  edgeThresh2: 2.11
  simThresh2: 0.812
  pixelRatio2: 0.067
  ```

---

## 💡 Claude Code 사용 팁

### 효율적인 사용법
1. **파일 직접 수정**: `optimization.py` 같은 대용량 파일
2. **반복 실험**: 파라미터 튜닝, 테스트 실행
3. **로그 분석**: 결과 파싱, 시각화

### Claude Chat이 나은 경우
1. **복잡한 디버깅**: 전체 구조 파악 필요
2. **알고리즘 설계**: 새로운 접근법 구상
3. **문서 작성**: README, 논문 작성

---

## 📝 Git 워크플로우

```bash
# 작업 시작
git pull origin main

# 수정 후 커밋
git add -A
git commit -m "[TYPE] Description"
# TYPE: FEAT, FIX, REFACTOR, TEST, DOC

# 푸시
git push origin main

# 태그 (마일스톤)
git tag -a v1.0 -m "BoRisk implementation complete"
git push --tags
```

---

## 🔄 컨텍스트 유지 전략

### 새 세션 시작시
```markdown
## Context
- Working on: BoRisk CVaR optimization
- Dataset: 119 welding images
- Current issue: [구체적 문제]
- Last result: CVaR=0.812
- Next step: [다음 목표]
```

### 주요 파일 해시 (변경 추적용)
```bash
# 현재 상태 저장
find . -name "*.py" -exec md5sum {} \; > file_hashes.txt

# 변경 확인
md5sum -c file_hashes.txt
```

---

## 📈 실험 추적

### 실험 로그 형식
```json
{
  "experiment_id": "exp_20241219_001",
  "config": {
    "iterations": 20,
    "n_initial": 15,
    "alpha": 0.3
  },
  "results": {
    "best_cvar": 0.812,
    "improvement": 8.3,
    "time_elapsed": 320.5
  },
  "notes": "Added GP normalization"
}
```

---

## 🎓 논문 작성용 정보

### 핵심 기여
1. BoRisk 알고리즘의 용접 라인 검출 적용
2. 10D 파라미터-환경 공간 최적화
3. CVaR 기반 강건성 확보
4. 실시간 처리 가능한 경량화

### 비교 대상
- Baseline: Grid Search
- Competitor 1: Standard BO (EI)
- Competitor 2: Random Search
- Ours: BoRisk with CVaR

---

## 📞 연락 및 협업

- GitHub: https://github.com/jam961020/graduate_master
- 주요 브랜치: main
- Issues: 버그 리포트 및 제안사항

---

마지막 업데이트: 2025.11.11
