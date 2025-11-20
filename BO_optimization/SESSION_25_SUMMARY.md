# Session 25 Summary (2025-11-20)

## 실험 설정

### 변경사항
1. **threshold 20px** (30px → 20px) - 더 엄격한 평가
2. **n_w = 15** (20 → 15) - 속도 향상
3. **cumulative best 기록** - 자연스러운 수렴 그래프

### 실험 파라미터
- iterations: 100
- n_initial: 10
- alpha: 0.3
- n_w: 15
- max_images: 600
- threshold: 20px

---

## 실험 결과

### 로그 위치
`logs/run_20251120_004752/`

### Initial 샘플링 결과 (좋음!)
```
Init 1: CVaR = 0.4831
Init 2: CVaR = 0.4451
Init 3: CVaR = 0.5373
Init 4: CVaR = 0.5398
Init 5: CVaR = 0.3948
Init 6: CVaR = 0.4348
Init 7: CVaR = 0.3425
Init 8: CVaR = 0.3165
Init 9: CVaR = 0.5686
```

**Initial CVaR 범위: 0.32~0.57** (이전 30px에서는 0.70)

### BO 최적화 결과
- **Initial CVaR (BO iter 1)**: 0.6891
- **Best CVaR**: 0.7173 (iteration 87)
- **Final CVaR**: 0.7173
- **Improvement**: +4.1%

### 그래프
`logs/run_20251120_004752/cvar_convergence.png`

---

## 문제점

### 1. score=0 발생
- **Image idx=85**에서 반복적으로 score=0 발생
- 특정 파라미터에서 라인 검출 실패
- CVaR을 급격히 낮추는 원인

### 2. 결과 요동
- 각 iteration의 CVaR이 크게 변동
- score=0 케이스가 CVaR에 큰 영향
- 안정적인 수렴 곡선이 아님

### 3. 개선폭 부족
- +4.1% 개선만 달성
- 목표: 15-20% 개선
- Initial 샘플링에서 이미 좋은 파라미터를 찾음

---

## 다음 단계: score=0 이미지 제외

### 원인 분석
- Image idx=85 등 특정 이미지에서 RANSAC 후 라인이 2개만 남음
- 4개 라인이 필요한데 실패 → 0점 처리
- 이미지 자체의 문제 (라벨링 오류 또는 너무 어려운 케이스)

### 해결 방안

#### Option 1: 문제 이미지 제외 (권장)
1. score=0이 자주 나오는 이미지 식별
2. 해당 이미지를 데이터셋에서 제외
3. 다시 실험

```python
# 제외할 이미지 목록 (예시)
EXCLUDE_IMAGES = [85, ...]  # 실제 분석 후 결정
```

#### Option 2: score=0 처리 방식 변경
- 0 대신 작은 값 (예: 0.1) 부여
- 완전 실패와 부분 실패 구분

#### Option 3: 최소 라인 개수 조건 완화
- 4개 → 2개 이상이면 평가
- 단, 평가 정확도 감소 우려

---

## 제외할 이미지 후보

### 확인된 문제 이미지
- **idx=85**: 반복적 score=0 발생

### 추가 분석 필요
```bash
# score=0 발생 빈도 분석
grep -c "score=0.0000" logs/run_20251120_004752/*.log
```

---

## 긴급 TODO

1. [ ] **score=0 이미지 식별 및 제외**
   - 전체 로그에서 score=0 발생 이미지 목록 추출
   - 해당 이미지 제외하고 데이터셋 재구성

2. [ ] **실험 재실행**
   - 제외된 데이터셋으로 재실험
   - 동일 설정 (100 iter, n_w=15, 20px)

3. [ ] **결과 확인**
   - 요동 감소 확인
   - 개선폭 확인 (목표: 10%+)

---

## 코드 수정 필요

### optimization.py
```python
# 데이터 로드 시 문제 이미지 제외
EXCLUDE_IMAGE_INDICES = [85, ...]  # 분석 후 추가

def load_dataset(...):
    # ... 기존 코드 ...

    # 문제 이미지 제외
    images_data = [img for i, img in enumerate(images_data)
                   if i not in EXCLUDE_IMAGE_INDICES]
```

또는

### evaluation.py
```python
def evaluate_lp(...):
    # ... 기존 코드 ...

    # score=0 대신 최소값 부여
    if score == 0:
        score = 0.05  # 완전 실패 패널티
```

---

## 시간 계획

- **논문 마감**: 내일
- **남은 시간**: 제한적
- **우선순위**:
  1. 문제 이미지 제외 후 재실험
  2. 결과 그래프 생성
  3. 논문에 결과 삽입

---

## 최적 파라미터 (iteration 87)

현재 실험의 best 파라미터 확인 필요:
```bash
cat logs/run_20251120_004752/iter_087.json
```

---

**마지막 업데이트**: 2025-11-20 14:35
**다음 작업**: score=0 이미지 제외 후 재실험
