# Claude Development Guide
## BoRisk CVaR Optimization for Welding Line Detection

Repository: https://github.com/jam961020/graduate_master

**최종 업데이트: 2025.11.12 23:45**
**✅ BoRisk KG 구현 완료 및 실험 진행 중**

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

### BoRisk 필수 구성 요소 (✅ 완료!)
1. ✅ 환경 벡터 추출 (`environment_independent.py`)
2. ✅ w_set 샘플링 (매 iteration마다)
3. ✅ GP 모델: (x, w) → y 학습 (15D 입력)
4. ✅ CVaR-KG 획득 함수 (`borisk_kg.py`)
5. ✅ CVaR objective 통합
6. ✅ 판타지 관측 구조 (SimplifiedBoRiskKG)
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

## 🎯 현재 작업 상태 (2025.11.12 23:45)

### ✅ 완료된 작업

#### 1. Repository Clone 및 경로 문제 해결 (완료 2025.11.12)
- Windows 로컬에 repository clone 완료
- `test_clone_final.py` 모든 하드코딩 경로 수정
- Windows → Linux 호환 경로로 변경 (`__file__` 기준)
- 주요 수정:
  - 카메라 파라미터 경로
  - YOLO/DexiNed 모델 경로
  - Windows 빌드 경로 주석 처리
  - samsung2024 디렉토리 체크 제거

#### 2. BoRisk 알고리즘 구현 완료 (완료 2025.11.12)
- BoRisk 논문 분석 및 완전 구현 ✅
- w_set 샘플링 시스템 구축 ✅
- GP 모델: (x, w) → y 학습 구조 ✅
- **CVaR-KG 획득 함수 적용 (`borisk_kg.py`)** ✅
- 환경 벡터 통합 완료 ✅
- `Simplified-CVaR-KG` 성공적으로 작동 확인 ✅

#### 3. 평가 메트릭 구현 (완료)
- **직선 방정식 기반 평가**
- `line_equation_evaluation()` (optimization.py:43-119)
- 방향 유사도 (60%) + 평행 거리 (40%)

#### 4. 9D 파라미터 최적화 (완료)
- AirLine 6D + RANSAC 3D
- `ransac_center_w`, `ransac_length_w`, `ransac_consensus_w`
- BOUNDS 업데이트 완료

#### 5. 환경 벡터 시스템 (완료)
- `environment_independent.py` 구현
- 6D 환경 특징 추출
- GP 입력으로 통합 완료

### 🔄 현재 상황 (2025.11.12 23:45)

#### 실행 환경
- **Windows 로컬**: 실행 가능 ✅
- **CRG311.pyd**: 설치 완료 ✅
- **코드 상태**: BoRisk KG 적용 완료 ✅

#### 실험 상태
- **BoRisk KG**: Simplified-CVaR-KG 성공 ✅
- **실험 진행 중**: alpha=0.1, 전체 데이터셋 (113장)
- **RANSAC 버그**: 1개 선 처리 로직 추가 완료 ✅
- **환경 벡터**: 6D 추출 성공, 분산 높음 (GP noise 0.74) ✅

### 🔴 긴급 해결 필요 문제점 (2025.11.12)

#### 1. 자동 라벨링 시스템 미구축 (최우선!)
**현재 상황**:
- 수동 라벨링에 의존 → 시간 소요 큼
- GT 데이터 부족 → 실험 속도 느림

**필요한 작업**:
- [ ] `auto_labeling.py` 작성
  - AirLine_assemble_test.py로 6개 점 자동 추출
  - ground_truth.json 포맷으로 저장
- [ ] `labeling_tool.py` 수정
  - 자동 생성 GT 불러오기 기능
  - 수동 수정 가능하도록

**우선순위**: 🚨 최우선 (오늘 중 완료 필요)

#### 2. 메트릭 재검토 필요 (High Priority)
**문제 관찰**:
- CVaR과 평균이 동일하게 움직임
- 실패 케이스를 제대로 구분하지 못하는 것으로 의심

**검토 사항**:
- [ ] 선이 검출 안 되는 경우: 0점 처리 적절한가?
- [ ] 방향은 맞는데 위치 틀린 경우: 거리 패널티 충분한가?
- [ ] 다양한 실패 케이스로 메트릭 테스트

**실험 필요**:
```python
test_cases = [
    ("완전 실패", detected=None),
    ("방향만 맞음", detected=parallel_but_far),
    ("위치만 맞음", detected=nearby_but_perpendicular),
]
```

#### 3. RANSAC 가중치 이해 오류 (High Priority)
**의심되는 문제**:
- Claude가 RANSAC 가중치를 잘못 이해했을 가능성
- 범위 [0.0, 1.0], [1, 10]이 적절한가?
- 정규화 제약 필요한가?

**수정 필요**:
- [ ] full_pipeline.py:160-236의 weighted_ransac_line 재검토
- [ ] 가중치 범위 재설정
- [ ] 극단값 실험으로 검증

#### 4. 환경 변수 효과 검증 (Medium Priority)
**의문**:
- n_w=15가 적절한가?
- 환경 샘플링이 diverse한가?
- alpha=0.3이 적절한가?

**실험 계획**:
- [ ] n_w: [10, 15, 20, 30] 비교
- [ ] alpha: [0.1, 0.2, 0.3, 0.4, 0.5] 비교
- [ ] 환경 샘플링 방식: 랜덤 vs k-means clustering

#### 5. BoRisk 이론 재확인 필요 (Medium Priority)
**질문**: 현재 구현이 BoRisk 논문과 일치하는가?
- 실제 이미지 평가 vs GP posterior 샘플링
- w_set 사용 방식이 맞는가?

**확인 필요**:
- [ ] BoRisk 논문 Algorithm 1 다시 확인
- [ ] BoTorch 튜토리얼과 비교
- [ ] 필요시 수정

### 📋 긴급 작업 우선순위 (2025.11.12)

#### Priority 1: 자동 라벨링 시스템 (최우선! 🚨)
**목표**: AirLine 결과로 GT 자동 생성
1. **auto_labeling.py 작성**
   - AirLine_assemble_test.py 사용
   - 6개 점 (longi 4개 + collar 2개) 자동 추출
   - ground_truth.json 포맷으로 저장

2. **labeling_tool.py 수정**
   - 자동 생성 GT 불러오기
   - 수동 수정 인터페이스

**예상 소요**: 1-2시간
**마감**: 오늘 중

#### Priority 2: 환경/alpha 실험 (High)
**목표**: 최적 하이퍼파라미터 찾기
1. **alpha 실험**
   ```bash
   python optimization.py --alpha 0.1 --iterations 10
   python optimization.py --alpha 0.2 --iterations 10
   python optimization.py --alpha 0.3 --iterations 10
   python optimization.py --alpha 0.4 --iterations 10
   python optimization.py --alpha 0.5 --iterations 10
   ```

2. **n_w 실험**
   ```bash
   python optimization.py --n_w 10 --iterations 10
   python optimization.py --n_w 15 --iterations 10
   python optimization.py --n_w 20 --iterations 10
   python optimization.py --n_w 30 --iterations 10
   ```

**예상 소요**: 실험당 30분-1시간
**마감**: 오늘 저녁

#### Priority 3: 시각화 및 분석 (High)
**목표**: 논문 Figure 생성
1. **visualization.py 작성**
   - 초기/중간/최종 선 검출 결과
   - CVaR 개선 추이 그래프
   - alpha별 성능 비교

2. **분석**
   - CVaR vs Mean 히스토그램
   - 환경별 성능 분포
   - 실패 케이스 분석

**예상 소요**: 2-3시간
**마감**: 오늘 밤

#### Priority 4: 메트릭 검증 (Medium)
**목표**: 현재 메트릭 문제 확인
1. **테스트 케이스 작성**
   - 다양한 실패 상황 시뮬레이션
   - 메트릭 응답 확인

2. **필요시 수정**
   - 거리 패널티 조정
   - 방향 가중치 조정

**예상 소요**: 1시간
**우선순위**: 시간 있으면

#### Priority 5: RANSAC 가중치 재검토 (Low)
**목표**: 가중치 이해 오류 확인
- full_pipeline.py 재검토
- 필요시 수정

**우선순위**: 나중에 (시간 있으면)

---

## 🚀 빠른 실행 명령어

### Windows 로컬 환경 (현재)
- 경로: `C:\Users\user\Desktop\study\task\graduate\graduate_master\BO_optimization`
- Python: 3.12.0 (weld2024_mk2 conda 환경)
- 상태: 실행 가능 ✓
- 데이터셋: `dataset/images/test/` (119장)
- 주의: 코드 복붙 상태 (정리 필요)

```bash
# conda 환경 활성화
conda activate weld2024_mk2

# 작업 디렉토리
cd C:/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

# 빠른 테스트 (2개 이미지)
python optimization.py --iterations 1 --n_initial 1 --alpha 0.3 --max_images 2

# alpha 실험 (각 10 iterations)
python optimization.py --alpha 0.1 --iterations 10 --n_initial 5
python optimization.py --alpha 0.2 --iterations 10 --n_initial 5
python optimization.py --alpha 0.3 --iterations 10 --n_initial 5
python optimization.py --alpha 0.4 --iterations 10 --n_initial 5
python optimization.py --alpha 0.5 --iterations 10 --n_initial 5

# n_w 실험
python optimization.py --n_w 10 --iterations 10 --n_initial 5
python optimization.py --n_w 15 --iterations 10 --n_initial 5
python optimization.py --n_w 20 --iterations 10 --n_initial 5
python optimization.py --n_w 30 --iterations 10 --n_initial 5

# 전체 실험 (최종)
python optimization.py --iterations 20 --n_initial 10 --alpha 0.3

# 결과 확인
dir results\
type logs\iter_*.json
```

### Linux 워크스테이션 (포기)
- 상태: Segmentation fault ❌
- 원인: CRG311 라이브러리 의존성 문제
- 결정: Windows로 회귀

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

**마지막 업데이트: 2025.11.12 22:35**
**🚨 긴급 상황: 오늘까지 실험 결과 필요**
**다음 세션 시작 시 반드시 NEXT_SESSION.md를 먼저 읽으세요!**
**최우선 작업: 자동 라벨링 시스템 구축!**

---

## 🚨 긴급 메모 (2025.11.12)

### 현재 상황
- **마감**: 오늘 (졸업 마감)
- **환경**: Windows 로컬 (Linux 포기)
- **상태**: 기본 실행 가능, 분석 및 개선 필요

### 오늘 해야 할 일 (우선순위)
1. ✅ **자동 라벨링** (auto_labeling.py)
2. ✅ **실험 돌리기** (alpha, n_w 조합)
3. ✅ **시각화** (Figure 생성)
4. ✅ **분석** (CVaR vs Mean, 메트릭 검증)

### 성공 기준
- [ ] 자동 라벨링 시스템 완성
- [ ] 최소 5개 실험 조합 완료
- [ ] 논문용 Figure 3개 이상
- [ ] 결과 분석 완료

**화이팅! 졸업하자! 🎓**
