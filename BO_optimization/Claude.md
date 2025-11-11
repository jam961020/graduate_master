# Claude Development Guide
## BoRisk CVaR Optimization for Welding Line Detection

Repository: https://github.com/jam961020/graduate_master

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

## 🎯 현재 작업 상태 (2025.11.11)

### 완료된 작업
- ✅ BoRisk 논문 구현 (10D 최적화)
- ✅ 한 스텝 = 한 평가 구조 수정
- ✅ CVaR GP 예측 구현
- ✅ AirLine 로깅 제거 (monkey patching)
- ✅ 환경 특징 자동 추출 (4D)

### 진행중 작업
- 🔄 획득 함수 튜닝 (CVaR Knowledge Gradient)
- 🔄 GP 하이퍼파라미터 최적화
- 🔄 평가 메트릭 가중치 조정

### 예정 작업
- 📋 RANSAC 가중치 파라미터 추가 (6D → 8D)
- 📋 CLIP 기반 환경 표현 (4D → latent)
- 📋 Multi-fidelity BO 구현

---

## 🚀 빠른 실행 명령어

```bash
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

### 1. AirLine 로깅 과다
```python
# 해결: monkey patching
full_pipeline.detect_lines_in_roi = quiet_detect_lines_in_roi
```

### 2. GP 학습 실패
```python
# 해결: Y 정규화
Y_normalized = (Y - Y.mean()) / (Y.std() + 1e-6)
```

### 3. 획득 함수 0 반환
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
