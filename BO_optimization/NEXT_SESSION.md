# 🚨 긴급 세션 가이드 - 2025-11-12

**상황**: 오늘까지 실험 결과를 내지 못하면 졸업 불가
**환경**: Windows 로컬 (리눅스 segfault로 회귀, 코드 복붙 사용 중)
**현재 상태**: 실험은 실행되나 결과 분석 및 개선 필요

---

## ✅ 완료된 작업 (2025.11.12 세션)

### 1. Repository Clone 및 경로 수정 완료 ✓
- 위치: `C:\Users\user\Desktop\study\task\graduate\graduate_master`
- `test_clone_final.py` 모든 하드코딩 경로 수정
- Windows 경로 → `__file__` 기준 절대 경로로 변경

### 2. BoRisk KG 구현 완료 ✓
- `borisk_kg.py` 추가: CVaR-KG 획득 함수
- `optimization.py` 수정: borisk_kg 통합
- `Simplified-CVaR-KG` 성공적으로 작동
- qMFKG 문제 해결

### 3. RANSAC 버그 수정 ✓
- `full_pipeline.py`: 1개 선만 검출된 경우 처리 로직 추가
- weighted_ransac_line 안정화

### 4. 실험 분석 완료 ✓
- GP noise level 의미 파악 (0.74 = 높음)
- CVaR vs Mean 차이 분석 (alpha=0.3: 71.7% vs 91.8%)
- alpha 조정 필요성 확인 → alpha=0.1로 실험 시작

---

## 🎯 다음 세션 작업 (우선순위)

### ✅ 현재 진행 중
- **실험 실행 중**: alpha=0.1, iterations=15, n_w=15, 전체 데이터셋
- **예상 완료**: 30분~1시간
- **결과 파일**: `results/bo_cvar_*.json`

### Priority 1: 실험 결과 분석 (최우선!)

**목표**: alpha=0.1 실험 결과 분석 및 추가 실험 계획

**작업**:
1. 실험 결과 확인
   ```bash
   # 결과 파일 확인
   ls -lt results/ | head -5

   # 최신 결과 보기
   cat results/bo_cvar_*.json | tail -1
   ```

2. CVaR 개선도 분석
   - 초기 CVaR vs 최종 CVaR
   - alpha=0.1이 극단값에 집중했는가?
   - 49%, 66% 같은 실패 케이스가 개선되었는가?

3. 추가 실험 결정
   - alpha=0.15, 0.2도 실험할지 결정
   - n_w 조정 필요성 판단

### Priority 2: 시각화 및 결과 정리 (High)

**목표**: 논문용 Figure 및 분석 자료 생성

**작업**:
1. **visualization.py 작성**
   - CVaR 개선 추이 그래프
   - alpha별 성능 비교 (0.1 vs 0.3)
   - 실패 케이스 분석

2. **결과 요약 문서**
   - 핵심 발견사항 정리
   - 논문용 Table 생성

### Priority 3: 추가 실험 (Medium)

**다음 실험 후보**:
```bash
# alpha=0.15 (중간값)
python optimization.py --alpha 0.15 --iterations 15 --n_initial 5 --n_w 15 --image_dir "../dataset/images/test" --gt_file "../dataset/ground_truth.json"

# alpha=0.2 (비교용)
python optimization.py --alpha 0.2 --iterations 15 --n_initial 5 --n_w 15 --image_dir "../dataset/images/test" --gt_file "../dataset/ground_truth.json"
```

### Priority 4: 자동 라벨링 시스템 구축 (이전 우선순위)

**목표**: AirLine_assemble_test.py 결과로 GT 자동 생성

#### 작업 단계:
1. **자동 라벨링 스크립트 작성**
   ```python
   # auto_labeling.py 생성
   # AirLine_assemble_test.py 사용하여 6개 점 추출
   # ground_truth.json 포맷으로 저장
   ```

2. **출력 포맷**
   ```json
   {
     "image_name": {
       "coordinates": {
         "longi_left_lower_x": 0, "longi_left_lower_y": 0,
         "longi_right_lower_x": 0, "longi_right_lower_y": 0,
         "longi_left_upper_x": 0, "longi_left_upper_y": 0,
         "longi_right_upper_x": 0, "longi_right_upper_y": 0,
         "collar_left_lower_x": 0, "collar_left_lower_y": 0,
         "collar_left_upper_x": 0, "collar_left_upper_y": 0
       }
     }
   }
   ```

3. **labeling_tool.py 연동**
   - 자동 생성된 GT를 수동으로 수정 가능하게
   - 기존 labeling_tool.py에 불러오기 기능 추가

#### 구현 위치:
```
BO_optimization/
├── auto_labeling.py          # 새로 생성
├── labeling_tool.py           # 기존 파일 수정
└── dataset/
    ├── ground_truth.json      # 기존
    └── ground_truth_auto.json # 자동 생성
```

---

### Priority 2: 환경 변수 조정 실험

**현재 문제**: 환경 벡터가 최적화에 제대로 반영되는가?

#### 실험 계획:
1. **환경 샘플링 방식 변경**
   - 현재: 랜덤 샘플링
   - 개선: Diverse sampling (k-means clustering)

2. **n_w 값 조정**
   ```bash
   # 현재: n_w=15
   python optimization.py --n_w 10 --iterations 10
   python optimization.py --n_w 20 --iterations 10
   python optimization.py --n_w 30 --iterations 10
   ```

3. **alpha 값 실험**
   ```bash
   # 현재: alpha=0.3 (worst 30%)
   python optimization.py --alpha 0.2  # worst 20%
   python optimization.py --alpha 0.4  # worst 40%
   python optimization.py --alpha 0.5  # worst 50%
   ```

---

### Priority 3: RANSAC 가중치 수정

**문제 발견**: Claude가 RANSAC 가중치를 잘못 이해한 듯

#### 현재 구현 (optimization.py:332-340):
```python
w_center = float(params.get('ransac_center_w', 0.5))
w_length = float(params.get('ransac_length_w', 0.5))
w_consensus = int(params.get('ransac_consensus_w', 5))
```

#### 수정 필요 사항:
1. **가중치 범위 재검토**
   - `ransac_center_w`: [0.0, 1.0] → 적절한가?
   - `ransac_length_w`: [0.0, 1.0] → 적절한가?
   - `ransac_consensus_w`: [1, 10] → 적절한가?

2. **가중치 정규화**
   - center + length = 1.0 제약 필요한가?
   - consensus는 곱셈 가중치로 사용

3. **실험**
   ```bash
   # 극단적인 값으로 테스트
   # center 중시
   python optimization.py --iterations 5 --n_initial 3

   # length 중시
   python optimization.py --iterations 5 --n_initial 3
   ```

---

### Priority 4: 시각화 - 초기/중간/최종 선 검출 결과

**목표**: 최적화 과정 시각화로 논문 Figure 생성

#### 필요한 시각화:
1. **초기 (iteration 0)**
   - 검출된 선
   - GT 선
   - 평가 점수

2. **중간 (iteration 10)**
   - 검출된 선
   - GT 선
   - 평가 점수
   - 개선 추이

3. **최종 (best iteration)**
   - 검출된 선
   - GT 선
   - 평가 점수
   - 최종 개선율

#### 구현:
```python
# visualization.py 생성
def save_detection_comparison(iteration, params, detected, gt, score):
    # 3개 subplot: 초기 / 중간 / 최종
    # 선 검출 결과 오버레이
    # 점수 표시
```

---

## 🔍 분석 포인트 (중요!)

### 1. 메트릭 재검토 필요

**의문**: 현재 메트릭이 실패 상황을 제대로 반영하는가?

#### 현재 메트릭 (line_equation_evaluation):
- 방향 유사도 (60%)
- 평행 거리 (40%)

#### 검토 사항:
- 선이 아예 검출 안 되면? → 0점 처리 맞나?
- 방향은 맞는데 위치가 크게 틀리면? → 거리 패널티 충분한가?
- GT가 없는 경우는? → 현재 skip

#### 실험:
```python
# 다양한 실패 케이스로 메트릭 테스트
test_cases = [
    ("완전 실패", detected=None, expected_score=0.0),
    ("방향만 맞음", detected=parallel_but_far, expected_score=?),
    ("위치만 맞음", detected=nearby_but_perpendicular, expected_score=?),
]
```

---

### 2. CVaR vs 평균 분석

**관찰**: 평균이 CVaR을 그대로 추종한다

#### 가설:
1. **데이터셋이 균질적이다**
   - 환경 변화가 크지 않음
   - 모든 이미지가 비슷한 난이도

2. **alpha가 너무 크다** (0.3)
   - worst 30% → 샘플이 많아서 평균과 비슷
   - alpha를 줄여서 극단치만 보면 차이가 날 수도

3. **메트릭 문제**
   - 실패 케이스를 제대로 구분 못함
   - 모든 이미지가 비슷한 점수대

#### 실험:
```bash
# alpha 조정 실험
python optimization.py --alpha 0.1 --iterations 10  # worst 10%
python optimization.py --alpha 0.5 --iterations 10  # worst 50%

# 결과 비교
# - CVaR vs Mean 차이 분석
# - 히스토그램 그려보기
```

---

### 3. GP 샘플링 vs 실제 평가

**BoRisk 핵심**: 실제 데이터를 쓰는 게 아니라 GP 샘플링 사용

#### 현재 구현:
- w_set 샘플링: 실제 이미지 인덱스 사용
- 평가: 실제 이미지로 평가

#### BoRisk 이론:
- w_set 샘플링: 환경 벡터만 샘플링
- 평가: **GP posterior에서 샘플링** (실제 평가 아님!)
- 장점: 실제로 없는 환경도 테스트 가능

#### 수정 필요 여부 검토:
```python
# 현재 (optimization.py:283-327)
def evaluate_on_w_set(X, images_data, yolo_detector, w_indices):
    # 실제 이미지 평가
    for idx in w_indices:
        img_data = images_data[idx]
        score = detect_and_evaluate(img_data)
```

**질문**: 이게 맞나? BoRisk 논문 다시 확인 필요

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

**제안**:
```bash
# Windows 브랜치 생성
git checkout -b windows-local

# Linux 수정사항 선택적으로 merge
git cherry-pick <경로 수정 커밋들>

# 또는 이번 세션 md만 업데이트
git checkout main
# NEXT_SESSION.md, Claude.md만 수정
git add *.md
git commit -m "docs: Update session guide with urgent tasks"
git push origin main
```

---

## 🚀 빠른 시작 명령어

### 환경 설정
```bash
# conda 환경 활성화
conda activate weld2024_mk2

# 작업 디렉토리
cd C:/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
```

### 1. 자동 라벨링 (최우선)
```bash
# 아직 없음 - 이번 세션에서 작성 필요
python auto_labeling.py --image_dir dataset/images/test --output dataset/ground_truth_auto.json
```

### 2. 빠른 실험
```bash
# 환경 변수 실험
python optimization.py --n_w 20 --alpha 0.2 --iterations 5 --n_initial 3

# RANSAC 가중치 실험
python optimization.py --iterations 5 --n_initial 3
```

### 3. 전체 실험
```bash
# 최종 실험
python optimization.py --iterations 20 --n_initial 10 --alpha 0.3
```

### 4. 시각화
```bash
# 아직 없음 - 이번 세션에서 작성 필요
python visualization.py --results results/bo_cvar_*.json
```

---

## 📊 성공 기준

### 오늘 달성해야 할 것:
1. ✅ 자동 라벨링 시스템 완성
2. ✅ 다양한 alpha/n_w 조합 실험 (최소 5개)
3. ✅ 시각화 Figure 생성 (초기/중간/최종)
4. ✅ 메트릭 분석 및 문제점 파악
5. ✅ CVaR vs Mean 분석 결과

### 논문용 Figure:
- Figure 1: 최적화 과정 (초기 → 중간 → 최종)
- Figure 2: CVaR 개선 추이 그래프
- Figure 3: alpha별 성능 비교
- Figure 4: 환경별 성능 분석

---

## 💡 AirLine 저자들은 바보인 듯

**관찰된 문제점**:
1. Windows 경로 하드코딩
2. 상대 경로 가정 (재현성 낮음)
3. 문서화 부족
4. 의존성 관리 엉망

**우리의 개선**:
1. ✅ __file__ 기준 절대 경로
2. ✅ 환경 독립적인 코드
3. ✅ 상세한 문서화 (이 파일!)
4. 🔄 conda 환경 명세 (TODO)

---

## 📝 다음 세션 TODO

### 즉시 시작:
- [ ] auto_labeling.py 작성
- [ ] labeling_tool.py 수정
- [ ] visualization.py 작성

### 실험:
- [ ] alpha [0.1, 0.2, 0.3, 0.4, 0.5] 실험
- [ ] n_w [10, 15, 20, 30] 실험
- [ ] 메트릭 테스트 케이스 작성

### 분석:
- [ ] CVaR vs Mean 히스토그램
- [ ] 환경별 성능 분포
- [ ] 실패 케이스 분석

### 문서:
- [ ] 실험 결과 정리
- [ ] Figure 생성
- [ ] 논문 초안 작성

---

**마지막 업데이트**: 2025-11-12 23:45
**다음 세션**: 실험 결과 분석 및 시각화!
**Status**: ✅ BoRisk KG 구현 완료, 실험 진행 중

**화이팅! 졸업하자! 🎓**
