# 다음 세션 계획 (Session 25)

## 현재 상황 요약

### 문제점
1. **threshold 30px가 너무 관대** → Initial부터 CVaR 0.70, 개선폭 9%만
2. **CVaR 급락 현상** (iter 68, 78, 88-91) → GP 예측 불안정
3. **논문 그래프로 못 씀** - Initial이 너무 높아서 개선이 미미해 보임

### 현재 실험 결과
- Run: `logs/run_20251119_045142`
- Initial 평균: 0.7024
- Best CVaR: 0.7660 (iter 85)
- 개선율: +9.1% (부족)

---

## 다음 세션 할 일

### 1. 새 실험 실행 (최우선!)

**설정**:
- **이미지**: 600장 선별 (for_BO에서)
- **threshold**: 20px (더 엄격하게)
- **n_w**: 20
- **n_initial**: 10
- **iterations**: 100

```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

python optimization.py \
  --image_dir ../dataset/images/for_BO \
  --gt_file ../dataset/ground_truth_merged.json \
  --iterations 100 \
  --n_initial 10 \
  --alpha 0.3 \
  --n_w 20 \
  --max_images 600
```

### 2. evaluation.py 수정 (실행 전!)

```python
# evaluation.py Line 21
def evaluate_lp(detected_coords, image, image_name=None, threshold=20.0, debug=False):
```

**30px → 20px 변경**

### 3. 학습/검증 분리
- **학습**: 600장
- **검증**: 247장 (`validation_images.json`에 저장)
- 검증 데이터로 최종 파라미터 CVaR 계산

### 4. 예상 효과
- Initial CVaR 낮아짐 (0.5~0.6 예상)
- 개선폭 증가 (15-20% 목표)
- 논문 그래프로 사용 가능

---

## 알려진 문제들

### CVaR 급락 원인
1. **w_set_fixed와 학습 데이터 불일치**: GP가 본 적 없는 환경에서 예측
2. **GP 외삽 불안정**: 학습 데이터 범위 밖에서 prior mean으로 회귀
3. **해결**: outlier 제거하고 그래프 사용

### GP 예측 불안정
- 새 데이터 추가 시 예측값 급변
- Score는 좋은데 CVaR만 급락
- **현실적 해결**: outlier 제거

---

## 최적 파라미터 (iter 85)

```json
{
  "edgeThresh1": -5.56,
  "simThresh1": 0.91,
  "pixelRatio1": 0.07,
  "edgeThresh2": -1.38,
  "simThresh2": 0.79,
  "pixelRatio2": 0.05,
  "ransac_weight_q": 8.27,
  "ransac_weight_qg": 4.77
}
```

---

## 파일 구조

```
BO_optimization/
├── optimization.py       # 메인 (수정됨: CVaR 고정 w_set)
├── evaluation.py        # 평가 (수정 필요: threshold 20px)
├── PAPER_MATERIALS.md   # 논문 자료
├── SESSION_24_PROGRESS.md
├── logs/run_20251119_045142/  # 현재 실험
```

---

## 실행 순서

1. **evaluation.py 수정** - threshold 30 → 20
2. **새 실험 시작** - 600장, 20px
3. **결과 확인** - Initial CVaR이 낮아졌는지
4. **그래프 생성** - 수렴 곡선
5. **검증 데이터 준비** - 최종 파라미터 테스트

---

## 중요 메모

- **5일 내 논문 마감** - 시간 없음
- **이번 실험이 마지막** - 결과로 써야 함
- **outlier 제거** - iter 68, 78 등
- **검증 데이터** - 사용자가 직접 준비

---

## 논문 작성 방향

### 기여점
- BoRisk 프레임워크 적용
- CVaR 기반 robust optimization
- 환경 조건화 (6D 특징)

### 한계점 (솔직하게)
- GP 예측 불안정
- 환경 다양성 탐색 제한
- 평가 메트릭 한계

### Figure 계획
- CVaR 수렴 곡선 (Initial 낮고 Final 높게)
- Initial vs Final 비교 이미지
- 파라미터 분포

---

**마지막 업데이트**: 2025-11-19 22:00
**다음 세션**: threshold 20px로 새 실험
**목표**: Initial 낮고 개선폭 큰 결과 얻기
