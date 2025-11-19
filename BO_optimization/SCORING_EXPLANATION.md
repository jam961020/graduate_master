# LP_r 연속 점수 계산 방식 상세 설명

## 📊 현재 점수 산정 기준 (evaluation.py)

### 1단계: 선을 픽셀로 변환
```python
# GT 선 3개 → 각 선당 100개 픽셀 샘플링
GT 선들 → 총 300개 GT 픽셀

# 검출된 선 3개 → 각 선당 100개 픽셀 샘플링
검출 선들 → 총 300개 검출 픽셀
```

### 2단계: 거리 계산
```python
# 각 GT 픽셀에서 가장 가까운 검출 픽셀까지의 거리 계산
for gt_pixel in GT_pixels:
    min_distance = min(거리(gt_pixel, 모든 검출 픽셀))
```

**예시**:
- GT 픽셀 1: 가장 가까운 검출 픽셀까지 **5px**
- GT 픽셀 2: 가장 가까운 검출 픽셀까지 **15px**
- GT 픽셀 3: 가장 가까운 검출 픽셀까지 **25px**
- ...

### 3단계: 연속 점수 계산 (핵심!)
```python
threshold = 10.0  # 현재 설정

for gt_pixel in GT_pixels:
    distance = min_distance[gt_pixel]

    # 연속 점수 공식
    pixel_score = max(0.0, 1.0 - distance / threshold)

    # 예시:
    # distance = 0px   → score = 1.0 - 0/10   = 1.0  (완벽!)
    # distance = 5px   → score = 1.0 - 5/10   = 0.5  (절반)
    # distance = 10px  → score = 1.0 - 10/10  = 0.0  (경계)
    # distance = 15px  → score = 1.0 - 15/10  = -0.5 → 0.0 (실패)
    # distance = 100px → score = 0.0 (완전 실패)
```

### 4단계: 최종 LP_r 점수
```python
LP_r = 모든 GT 픽셀의 평균 점수

# 예시: 300개 GT 픽셀
LP_r = (pixel_score[0] + pixel_score[1] + ... + pixel_score[299]) / 300
```

---

## 🎯 Threshold의 의미

### Threshold = 10px (현재)
```
이미지 해상도: 2448 × 3264
10px = 가로의 0.41%, 세로의 0.31%

의미: "검출된 선이 GT에서 10px 이내에 있어야 점수를 받음"
```

**너무 엄격한 이유**:
- 2448px 폭 이미지에서 10px = 매우 작은 허용 오차
- Pixel-perfect에 가까운 수준 요구
- 조금만 어긋나도 점수가 급격히 하락

**거리별 점수 (10px threshold)**:
| 거리 | 점수 | 평가 |
|------|------|------|
| 0-5px | 0.5-1.0 | 매우 좋음 |
| 5-8px | 0.2-0.5 | 보통 |
| 8-10px | 0.0-0.2 | 거의 실패 |
| 10px+ | 0.0 | 완전 실패 |

### Threshold = 20px (제안)
```
20px = 가로의 0.82%, 세로의 0.61%

의미: "검출된 선이 GT에서 20px 이내에 있어야 점수를 받음"
```

**더 현실적인 이유**:
- 실제 용접선 검출에서 10-20px 오차는 허용 가능
- 연속 점수의 장점(부드러운 gradient)을 살림
- BO가 학습할 여지를 줌

**거리별 점수 (20px threshold)**:
| 거리 | 점수 | 평가 |
|------|------|------|
| 0-5px | 0.75-1.0 | 매우 좋음 |
| 5-10px | 0.5-0.75 | 좋음 |
| 10-15px | 0.25-0.5 | 보통 |
| 15-20px | 0.0-0.25 | 아쉬움 |
| 20px+ | 0.0 | 실패 |

### Threshold = 30px (더 관대한 옵션)
```
30px = 가로의 1.23%, 세로의 0.92%

의미: "검출된 선이 GT에서 30px 이내에 있어야 점수를 받음"
```

**가장 관대한 옵션**:
- 초기 학습에 유리
- CVaR이 빠르게 올라감
- 나중에 threshold 줄여서 fine-tuning 가능

---

## ⚠️ Score = 0이 나오는 경우

### 케이스 1: 선이 아예 검출 안 됨
```python
detected_lines = []  # 빈 리스트

→ detected_pixels = []
→ LP_r = 0.0 (즉시 반환)
```

**원인**:
- AirLine 파라미터가 너무 엄격
- 이미지가 너무 어려움 (노이즈, 낮은 대비)

### 케이스 2: 선은 검출됐지만 GT에서 너무 멀리 떨어짐
```python
모든 GT 픽셀의 min_distance > threshold (10px)

예시:
- GT 픽셀들의 거리: [15px, 20px, 12px, 18px, ...]
- threshold = 10px
- 모든 픽셀 점수 = 0.0
- LP_r = 0.0
```

**원인**:
- Threshold가 너무 엄격 (10px)
- 검출은 됐지만 위치가 조금 어긋남

---

## 📈 현재 실험 문제 분석

### 실험 결과 (10px threshold, 11 iterations)
```
Score = 0 발생 비율: 72.7% (8/11)

자주 실패하는 이미지:
- 이미지 279: 6회 연속 score=0
- 이미지 278, 286: score=0
```

### 문제 1: Threshold가 너무 엄격
**10px는 너무 작음**:
- 연속 점수를 쓰는 의미가 없음
- 대부분의 검출 결과가 10px 밖으로 나감
- → Score = 0

### 문제 2: 특정 이미지가 너무 어려움
**이미지 279 분석 필요**:
- 왜 계속 검출 실패하는가?
- 노이즈? 낮은 대비? 복잡한 패턴?
- → 데이터 품질 문제 가능성

---

## 💡 해결 방안

### Option 1: Threshold 증가 (추천!)
```python
# evaluation.py Line 246
threshold = 20.0  # 10.0 → 20.0

또는

threshold = 30.0  # 더 관대하게
```

**예상 효과**:
- Score=0 비율 감소 (72% → 20% 이하)
- 연속 점수의 장점 활용
- BO가 학습할 여지 생김
- CVaR 개선 가능성 증가

### Option 2: 문제 이미지 제외
```python
# optimization.py에서
bad_images = [278, 279, 286]  # score=0 자주 나오는 이미지
filter images not in bad_images
```

**장점**: 학습 효율 증가
**단점**: 데이터셋 감소, 일반화 저하

### Option 3: 점진적 Threshold 감소
```python
# 1단계: threshold=30으로 빠른 학습
# 2단계: threshold=20으로 fine-tuning
# 3단계: threshold=10으로 최종 정밀화
```

---

## 🎯 권장 다음 실험

### 실험 A: 20px threshold (추천!)
```bash
# evaluation.py 수정
threshold = 20.0

# 실행
python optimization.py --iterations 100 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json
```

**예상**:
- Score=0 비율: 20-30%로 감소
- CVaR: 0.5-0.7 달성 가능
- 학습 곡선: 점진적 개선

### 실험 B: 30px threshold (빠른 검증)
```bash
threshold = 30.0

python optimization.py --iterations 50 --n_initial 10 --alpha 0.3 \
  --gt_file ../dataset/ground_truth_auto.json
```

**예상**:
- Score=0 비율: 10% 이하
- CVaR: 0.6-0.8 달성 가능
- 빠른 수렴

---

## 📊 비교 요약

| Threshold | 엄격도 | Score=0 예상 | CVaR 예상 | 학습 가능성 | 추천도 |
|-----------|--------|--------------|-----------|-------------|--------|
| **10px** | 매우 엄격 | 70%+ | 0.3대 | 낮음 | ❌ |
| **20px** | 적당 | 20-30% | 0.5-0.7 | 높음 | ✅✅✅ |
| **30px** | 관대 | 10% | 0.6-0.8 | 매우 높음 | ✅✅ |

---

**결론**:
1. 연속 점수를 쓴다면 threshold는 **20px 이상**이 적절
2. 10px는 너무 엄격해서 연속성의 장점을 못 살림
3. 먼저 20px로 실험, 성공하면 15px로 fine-tuning 가능

**핵심**: Threshold가 엄격할수록 좋은 게 아님!
→ BO가 학습할 여지를 줘야 개선 가능!
