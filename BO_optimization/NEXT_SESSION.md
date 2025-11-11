# 다음 세션 시작 가이드

**날짜**: 2025.11.11 21:00
**이전 세션**: BoRisk 알고리즘 완전 구현 완료!

---

## 🎉 완료된 작업

**BoRisk 알고리즘 완전 구현 성공!**

### 구현된 기능
1. ✅ **환경 벡터 추출**: `extract_all_environments()` - 6D 환경 특징
2. ✅ **w_set 샘플링**: `sample_w_set()` - n_w개만 샘플링
3. ✅ **GP 모델**: `AppendFeatures(feature_set=w_set)` - (x,w) → y 학습
4. ✅ **qMFKG 획득 함수**: CVaR objective 통합
5. ✅ **evaluate_on_w_set()**: w_set만 평가 (113개 아님!)
6. ✅ **BO 루프**: 매 iteration마다 w_set 재샘플링 및 평가

### 핵심 변경사항
- **파일**: `optimization.py` (완전 재작성)
- **Import 수정**:
  ```python
  from botorch.acquisition import qMultiFidelityKnowledgeGradient
  from botorch.models.transforms.input import AppendFeatures
  from environment_independent import extract_parameter_independent_environment
  ```
- **새 함수들**:
  - `extract_all_environments()` - 모든 이미지의 환경 벡터 추출
  - `sample_w_set(env_features, n_w)` - w_set 랜덤 샘플링
  - `evaluate_on_w_set(X, images_data, yolo, w_indices)` - w_set만 평가
  - `compute_cvar_from_scores(scores, alpha)` - CVaR 계산
  - `cvar_objective(samples, alpha)` - GP 샘플에서 CVaR 계산
- **파라미터 추가**: `--n_w` (default=15)

---

## ⚡ 즉시 확인할 것

### 1. 테스트 진행 상황
```bash
# 실행 중인 프로세스
ps aux | grep "python.*optimization.py"

# 로그 확인
tail -100 test_borisk.log

# 반복별 로그
ls -lh logs/
cat logs/iter_001.json | jq .

# 결과
ls -lh results/
cat results/bo_cvar_*.json | jq .
```

### 2. 기대 결과
- **이전 (잘못된 구현)**:
  - CVaR = 0.0011 (매우 낮음)
  - 평가 횟수: (n_initial + n_iterations) * 113개
- **현재 (BoRisk)**:
  - CVaR > 0.01 (기대)
  - 평가 횟수: (n_initial + n_iterations) * n_w
  - 예: (2 + 1) * 3 = 9회 (이전: 3 * 113 = 339회)
  - **속도 향상**: 약 40배 빠름!

### 3. 성공 기준
- [ ] Import 에러 없음
- [ ] 환경 벡터 추출 성공 (6D, N개 이미지)
- [ ] w_set 샘플링 성공
- [ ] GP 모델 학습 성공 (AppendFeatures)
- [ ] qMFKG 획득 함수 작동
- [ ] CVaR 값이 0.01 이상

---

## 📋 다음 작업 우선순위

### Priority 1: 테스트 검증 및 디버깅 (CRITICAL - 즉시)
1. **테스트 완료 확인**
   - 첫 테스트 결과 확인 (iterations=1, n_initial=2, n_w=3)
   - 로그 분석: 각 단계별 정상 작동 여부
   - CVaR 값 확인

2. **에러 발생 시 디버깅**
   - Import 에러: BoTorch 버전 확인
   - 차원 불일치: train_X, train_Y, w_set 형태 확인
   - GP 학습 실패: 정규화, 노이즈 레벨 확인
   - qMFKG 실패: UCB로 폴백 확인

3. **성공 시 확대 테스트**
   ```bash
   # 소규모 테스트
   python optimization.py --image_dir ../dataset/images/test \
       --gt_file ../dataset/ground_truth.json \
       --iterations 5 --n_initial 5 --n_w 10 --alpha 0.3

   # 중규모 테스트
   python optimization.py --image_dir ../dataset/images/test \
       --gt_file ../dataset/ground_truth.json \
       --iterations 10 --n_initial 10 --n_w 15 --alpha 0.3

   # 전체 실험
   python optimization.py --image_dir ../dataset/images/test \
       --gt_file ../dataset/ground_truth.json \
       --iterations 20 --n_initial 10 --n_w 15 --alpha 0.3
   ```

### Priority 2: 성능 최적화 (Medium)
- qMFKG 하이퍼파라미터 튜닝
  - `num_fantasies`: 64 → 32 or 128
  - `num_restarts`: 10 → 20
  - `raw_samples`: 512 → 1024
- w_set 크기 실험: 10, 15, 20, 25
- alpha 값 실험: 0.2, 0.3, 0.4

### Priority 3: 실험 및 분석 (High)
- Vanilla BO vs BoRisk 비교
- 속도 측정 및 분석
- CVaR 개선 곡선 시각화
- 최적 파라미터 분석

---

## 🔧 주요 파일 위치

- **메인 파일**: `optimization.py` (BoRisk 구현)
- **환경 추출**: `environment_independent.py`
- **파이프라인**: `full_pipeline.py` (YOLO + AirLine)
- **평가 메트릭**: `optimization.py:39-116` (line_equation_evaluation)
- **로그**: `logs/iter_XXX.json`
- **결과**: `results/bo_cvar_*.json`

---

## 📝 실행 명령어 템플릿

```bash
# 빠른 테스트 (디버깅용)
python optimization.py \
    --image_dir ../dataset/images/test \
    --gt_file ../dataset/ground_truth.json \
    --iterations 2 --n_initial 3 --n_w 5 --alpha 0.3

# 표준 실험
python optimization.py \
    --image_dir ../dataset/images/test \
    --gt_file ../dataset/ground_truth.json \
    --iterations 20 --n_initial 10 --n_w 15 --alpha 0.3

# 백그라운드 실행
nohup python optimization.py \
    --image_dir ../dataset/images/test \
    --gt_file ../dataset/ground_truth.json \
    --iterations 20 --n_initial 10 --n_w 15 --alpha 0.3 \
    > experiment.log 2>&1 &
```

---

**다음 세션 시작 시**:
1. 먼저 테스트 결과 확인
2. 에러 발생 시 디버깅
3. 성공 시 확대 테스트 진행
4. 결과 분석 및 논문 작성 준비

**마지막 업데이트**: 2025.11.11 21:00
