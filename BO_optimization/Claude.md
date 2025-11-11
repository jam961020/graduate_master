# Claude Development Guide
## BoRisk CVaR Optimization for Welding Line Detection

Repository: https://github.com/jam961020/graduate_master

**최종 업데이트: 2025.11.11 20:30**

## 📌 대전제

- **이 프로젝트의 핵심은 BoRisk 알고리즘의 올바른 구현**
- BoRisk = Risk-aware Bayesian Optimization using CVaR (Conditional Value at Risk)
- 논문: ["Bayesian Optimization under Risk" (BoRisk)](https://arxiv.org/abs/2011.05939)
- **⚠️ 하드코딩으로 우회하지 말고 문제의 본질을 해결하라**
- **⚠️ 임시 해결책 사용 시 반드시 TODO 주석을 남겨라**
- **⚠️ NEXT_SESSION.md 파일이 본 파일보다 더 자세한 다음 task를 담고있다. 읽고 시작하라**

## 🎯 BoRisk 알고리즘 핵심 원리 (완벽 이해 완료)

### 기본 구조
1. **초기화**: n_initial개 (x,w) 쌍을 실제로 평가
2. **매 iteration**: **단 1개 (x,w) 쌍만 실제 평가!** (113개 전체 아님!)
3. **CVaR 계산**: GP의 판타지 샘플링으로 계산 (실제 평가 아님!)
4. **w_set**: 획득 함수에서 판타지로만 사용 (10~20개 샘플)

### 현재 구현의 Critical 문제
**⚠️ 현재 코드는 BoRisk가 아닌 Vanilla BO + CVaR objective!**

| 항목 | 현재 (잘못됨) | BoRisk (올바름) |
|------|--------------|----------------|
| **평가 개수** | 매번 113개 전체 | 매번 n_w개 (10~20개) |
| **GP 모델** | x → y | (x, w) → y |
| **획득 함수** | EI/UCB | ρKG (qMFKG) |
| **CVaR 계산** | 직접 평가 | GP 샘플링 |
| **속도** | 매우 느림 | 빠름 (1/10) |

### BoRisk 필수 구성 요소 (현재 누락됨)
1. ✅ 환경 벡터 추출 (`environment_independent.py`)
2. ❌ w_set 샘플링 및 AppendFeatures
3. ❌ GP 모델: (x, w) → y 학습
4. ❌ qMultiFidelityKnowledgeGradient 획득 함수
5. ❌ CVaR objective 통합
6. ❌ 판타지 관측 구조
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

## 🎯 현재 작업 상태 (2025.11.11 20:30)

### ✅ 완료된 작업

#### 1. BoRisk 알고리즘 완벽 이해 (완료 20:05)
- BoRisk 논문 및 BoTorch 튜토리얼 분석 완료
- 핵심 원리 파악: 매 iteration 1개 (x,w) 쌍만 평가
- w_set 샘플링, GP 판타지, qMFKG 획득함수 구조 이해
- `optimization_borisk.py` 발견 (기존 구현 존재)

#### 2. CRG311 Linux 빌드 (완료 19:00)
- AirLine 공식 리포에서 C++ 소스 컴파일
- pybind11로 Linux .so 생성
- 경로 수정 및 lazy initialization 적용

#### 3. 평가 메트릭 변경 (완료 19:28)
- **끝점 기반 → 직선 방정식 기반**
- `line_equation_evaluation()` 함수 추가 (optimization.py:39-116)
- Ax + By + C = 0 형식으로 정규화
- 방향 유사도 (법선 벡터 내적) + 평행 거리
- 가중치: direction 60%, distance 40%

#### 4. RANSAC 가중치 최적화 (완료 19:28)
- **6D → 9D 확장**
- BOUNDS 업데이트: 9D [AirLine 6D + RANSAC 3D]
- `ransac_center_w`, `ransac_length_w`, `ransac_consensus_w` 추가
- Sobol 엔진 차원 수정: dimension=9
- objective_function에 파라미터 전달 구현

#### 5. 로깅 최적화 (완료 19:28)
- 화면 출력 최소화 (토큰 절약)
- 상세 로그를 `logs/iter_XXX.json`로 파일 저장
- 각 반복마다 9D 파라미터, CVaR, 획득함수 값 기록

#### 6. 환경 벡터 추출 구현 (완료)
- `environment_independent.py` - 6D 환경 특징 추출
- brightness, contrast, edge_density, texture_complexity, blur_level, noise_level

### 🔄 진행 중

- **이전 테스트 결과 확인 필요**
- 로그: `new_test.log` (초기 샘플링 단계에서 멈춤)
- 결과: `results/bo_cvar_20251111_191029.json` (19:10 실행)

### 🔴 Critical 문제점 - 최우선 해결 필요

#### 1. BoRisk 알고리즘 구조 완전 누락 (CRITICAL - 최우선)
**현재 코드는 BoRisk가 아니라 Vanilla BO + CVaR objective!**

**문제점**:
```python
# optimization.py:217-273 - 매 iteration마다 113개 이미지 전부 평가
def objective_function(X, images_data, yolo_detector, alpha=0.3):
    scores = []
    for img_data in images_data:  # 113개 전체 순회!
        score = line_equation_evaluation(...)
        scores.append(score)
    cvar = np.mean(np.sort(scores)[:n_worst])  # 직접 CVaR 계산
```

**BoRisk 올바른 방식**:
```python
# 1. w_set 샘플링 (10~20개만)
w_set, w_indices = sample_w_set(all_env_features, n_w=15)

# 2. GP 모델: (x, w) → y
model = SingleTaskGP(
    train_X,  # [N, 9] params만
    train_Y,  # [N*n_w, 1] 각 x마다 n_w개 환경
    input_transform=AppendFeatures(feature_set=w_set)
)

# 3. qMFKG 획득 함수 + CVaR objective
acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    num_fantasies=64,
    objective=GenericMCObjective(cvar_objective)
)

# 4. 매 iteration: n_w개만 평가 (15개, 113개 아님!)
candidate = optimize_acqf(acqf, bounds, q=1)
observations = evaluate_on_w_set(candidate, w_indices)  # 15개만!
```

**필요한 수정**:
- [ ] w_set 샘플링 함수 구현
- [ ] AppendFeatures input_transform 추가
- [ ] GP 모델 구조 변경: x → y에서 (x,w) → y로
- [ ] qMultiFidelityKnowledgeGradient 획득 함수 적용
- [ ] CVaR objective 통합
- [ ] evaluate_on_w_set 함수로 평가 방식 변경

#### 2. 환경 변수 미통합 (CRITICAL)
- `environment_independent.py` 구현되어 있으나 **optimization.py에서 전혀 사용 안 함**
- GP가 (x, z) → y 학습하지 않고 x → y만 학습
- BoRisk의 핵심인 환경 조건부 예측 누락
- TODO: 환경 벡터를 w로 사용하여 GP 입력 구성

#### 3. 평가 효율성 (CRITICAL)
- 매번 113개 이미지 전체 평가 → 매우 느림
- BoRisk는 매번 10~20개만 평가 → 10배 빠름
- 현재 구조로는 실험 불가능 (시간 초과)

### 📋 다음 작업 우선순위

#### Priority 1: BoRisk 알고리즘 구현 (Critical - 최우선)
1. **w_set 샘플링 시스템 구축**
   - 모든 이미지의 환경 벡터 사전 추출
   - sample_w_set() 함수 구현 (n_w=15개)
   - 인덱스 추적 시스템

2. **GP 모델 구조 변경**
   - AppendFeatures input_transform 적용
   - (x, w) → y 학습 구조로 변경
   - train_X: [N, 9], train_Y: [N*n_w, 1]

3. **qMFKG 획득 함수 구현**
   - qMultiFidelityKnowledgeGradient import
   - CVaR objective 함수 작성
   - 판타지 샘플링 설정

4. **평가 함수 분리**
   - evaluate_on_w_set() 함수 구현
   - objective_function()은 초기화 단계만 사용
   - BO 루프에서 w_set만 평가

#### Priority 2: 환경 벡터 통합 (High)
- environment_independent.py 연동
- 이미지별 환경 특징 추출 및 저장
- GP 입력으로 환경 벡터 사용

#### Priority 3: 실험 및 검증 (Medium)
- 소규모 테스트 (n_initial=5, iterations=10)
- CVaR 값 모니터링 및 개선 확인
- 전체 실험 실행

---

## 🚀 빠른 실행 명령어

### 워크스테이션 환경
- 경로: `/home/jeongho/projects/graduate/BO_optimization`
- Python: 3.11.14 (weld2024_mk2 환경)
- GPU: CUDA 12.4 available
- 데이터셋: `../dataset/images/test/` (113장 실제 사용)

```bash
# 디버그 테스트 (빠른 검증)
python optimization.py --iterations 2 --n_initial 3 --alpha 0.3

# 소규모 테스트 (BoRisk 검증용)
python optimization.py --iterations 10 --n_initial 5 --alpha 0.3

# 표준 실행 (20회)
python optimization.py --iterations 20 --n_initial 10 --alpha 0.3

# 전체 실행 (30회)
python optimization.py --iterations 30 --n_initial 15 --alpha 0.2

# 백그라운드 실행 (로그 저장)
nohup python optimization.py --iterations 20 --n_initial 10 --alpha 0.3 > experiment.log 2>&1 &

# 실행 상태 확인
tail -f experiment.log
ps aux | grep "python.*optimization.py"

# 결과 확인
ls -lh results/
cat logs/iter_*.json | tail -20
```

---

## 🔧 주요 파라미터

### 최적화 파라미터 (9D)

#### AirLine 파라미터 (6D)
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| edgeThresh1 | [-23.0, 7.0] | -3.0 | Q 프리셋 엣지 임계값 |
| simThresh1 | [0.5, 0.99] | 0.98 | Q 프리셋 유사도 |
| pixelRatio1 | [0.01, 0.15] | 0.05 | Q 프리셋 픽셀 비율 |
| edgeThresh2 | [-23.0, 7.0] | 1.0 | QG 프리셋 엣지 임계값 |
| simThresh2 | [0.5, 0.99] | 0.75 | QG 프리셋 유사도 |
| pixelRatio2 | [0.01, 0.15] | 0.05 | QG 프리셋 픽셀 비율 |

#### RANSAC 가중치 (3D)
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| ransac_center_w | [0.0, 1.0] | 0.5 | 중심 거리 가중치 |
| ransac_length_w | [0.0, 1.0] | 0.3 | 라인 길이 가중치 |
| ransac_consensus_w | [1, 10] | 5 | Consensus 가중치 |

### 환경 벡터 (6D) - w로 사용
| Feature | Range | Description |
|---------|-------|-------------|
| brightness | [0, 1] | 평균 밝기 (mean/255) |
| contrast | [0, 1] | 표준편차/128 |
| edge_density | [0, 1] | Canny 엣지 픽셀 비율 |
| texture_complexity | [0, 1] | Laplacian 분산 기반 |
| blur_level | [0, 1] | 블러 정도 |
| noise_level | [0, 1] | 노이즈 수준 |

### BoRisk 하이퍼파라미터
- `n_w`: w_set 크기 (기본값: 15)
- `num_fantasies`: 판타지 샘플 개수 (기본값: 64)
- `alpha`: CVaR threshold (기본값: 0.3, worst 30%)
- `n_initial`: 초기 샘플링 개수 (기본값: 10)

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
2. 15D 파라미터-환경 공간 최적화 (params 9D + env 6D)
3. CVaR 기반 강건성 확보
4. w_set 샘플링 기반 효율적 평가
5. 직선 방정식 기반 평가 메트릭

### 비교 대상
- Baseline: Grid Search
- Competitor 1: Standard BO (EI)
- Competitor 2: Random Search
- Ours: BoRisk with CVaR + qMFKG

### 주요 수식
- CVaR_α(f(x,w)) = E[f(x,w) | f(x,w) ≤ F^(-1)(α)]
- GP: f(x,w) ~ GP(μ, k((x,w), (x',w')))
- qMFKG with fantasy observations

---

## 📞 연락 및 협업

- GitHub: https://github.com/jam961020/graduate_master
- 주요 브랜치: main
- Issues: 버그 리포트 및 제안사항

---

**마지막 업데이트: 2025.11.11 20:30**
**다음 세션 시작 시 반드시 NEXT_SESSION.md를 먼저 읽으세요!**
