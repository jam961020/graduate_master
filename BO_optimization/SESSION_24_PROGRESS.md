# Session 24 Progress (2025-11-19)

## 현재 실험 상태

### 실행 중인 실험
- **Run**: `logs/run_20251119_045142`
- **Resume from**: checkpoint_iter_050
- **n_w**: 15 → 20으로 증가
- **목표**: 200 iterations

### 진행률
- iter 50까지 완료 (CVaR 0.7485)
- iter 51~54에서 CVaR 급락 발생 (0.42 → 0.37)
- 현재 n_w=20으로 다시 resume 중

---

## 발견된 문제들

### 1. 이미지 선택 다양성 부족
- 847개 이미지 중 13개만 선택됨 (1.5%)
- Image 37, 279가 반복적으로 선택
- **원인**: KG가 exploitation에 치우침

### 2. CVaR 급락 현상 (iter 53-54)
```
iter_52: CVaR=0.7485, Score=0.8932
iter_53: CVaR=0.4221, Score=0.9081  ← 급락!
iter_54: CVaR=0.3738, Score=0.9004
```

**현상**: Score는 좋은데 CVaR만 급락
**추정 원인**:
- Sobol 시드 변경 시도 → 완전히 다른 환경 샘플링
- GP가 새 환경에 대한 데이터 없어서 예측 불안정
- 시드를 원래대로 (`seed=iteration`) 복구함

### 3. 평가 메트릭 한계
- iter_051: 눈으로 "망한 수준"인데 Score=0.7716
- threshold=30px가 너무 관대함
- 실제 품질과 점수 간 괴리 존재

### 4. CVaR의 의미 문제
- CVaR이 진정한 "worst-case 환경 개선"인지 의문
- 같은 이미지만 반복 선택되면 특정 이미지에 과적합
- 용접 비드 등 어려운 케이스를 찾아서 개선하는 게 아님

---

## 주요 변경사항

### Sobol 시드 (원복됨)
```python
# optimization.py Line 833
w_set, w_indices = sample_w_set(all_env_features, n_w=n_w, seed=iteration)
```
- 시도: `seed=(iteration * 12347 + 5923) % 2^31`
- 결과: CVaR 급락 → 원복

### n_w 증가
- 15 → 20으로 증가
- 더 많은 환경 샘플링으로 다양성 개선 기대

---

## CVaR 추이

### Initial (1-10)
- 범위: 0.18 ~ 0.53
- 목표: 0.7+

### BO Progress
| Iter | CVaR | Score | Image |
|------|------|-------|-------|
| 36 | 0.7359 | 0.9177 | 279 |
| 45 | 0.7433 | 0.9293 | - |
| 50 | 0.7474 | 0.8962 | 279 |
| 52 | 0.7485 | 0.8932 | 37 |
| 53 | 0.4221 | 0.9081 | 37 |
| 54 | 0.3738 | 0.9004 | 51 |

**Best CVaR**: 0.7485 (iter 52)

---

## 논문 작성 방향

### 1. 기여점
- BoRisk 프레임워크를 용접선 검출에 적용
- CVaR 기반 robust 파라미터 최적화
- Initial → Final CVaR 개선 (0.3~0.5 → 0.74+)

### 2. 환경 특징 정당화
- Pearson 상관계수로 LP_r과 상관관계 분석
- 해석 가능한 6개 특징 선별 (brightness, contrast 등)
- "왜 CLIP 안 썼나?" → 상관분석 기반 선택

### 3. 한계점 (솔직하게 명시)
- 환경 다양성 탐색 제한 (같은 이미지 반복)
- 평가 메트릭의 한계 (soft scoring 관대함)
- CVaR 급락 현상 (GP 예측 불안정)

### 4. Future Work
- Exploration 강화 (UCB 계열 적용)
- 어려운 이미지 의도적 포함
- 평가 메트릭 개선 (threshold 조정)

---

## 다음 세션 할 일

### 1. 실험 결과 확인
```bash
# 최신 결과 확인
python -c "
import json, glob
files = sorted(glob.glob('logs/run_20251119_045142/iter_*.json'))
for f in files[-10:]:
    d = json.load(open(f))
    print(f'Iter {d[\"iteration\"]:3d}: CVaR={d[\"cvar\"]:.4f}')
"
```

### 2. Visualization 생성
```bash
python visualization_exploration.py logs/run_20251119_045142
```
- CVaR 수렴 곡선 (best_cvar_history 사용)
- 이상치 제거 필요 (iter 53-54)

### 3. 어려운 이미지 퀵 테스트
- 비드/노이즈 많은 이미지 선별
- 최적 파라미터로 테스트
- robust 성능 검증

### 4. 논문 초안 작성
- 5일 내 완성 필요
- Figure 우선 생성
- 한계점 솔직하게 명시

---

## 실행 명령어

### 진행 확인
```bash
ls -lt logs/run_20251119_045142/iter_*.json | head -5
```

### Resume (필요시)
```bash
python optimization.py \
  --image_dir ../dataset/images/for_BO \
  --gt_file ../dataset/ground_truth_merged.json \
  --iterations 150 \
  --alpha 0.3 \
  --n_w 20 \
  --resume_from logs/run_20251119_045142
```

### Visualization
```bash
python visualization_exploration.py logs/run_20251119_045142
```

---

## 중요 파일

- `optimization.py` - 메인 최적화 코드
- `evaluation.py` - 평가 메트릭 (threshold=30px)
- `visualization_exploration.py` - 시각화
- `logs/run_20251119_045142/` - 현재 실험 로그

---

## 핵심 메모

1. **Score와 CVaR 구분**: Score=실제평가, CVaR=GP예측
2. **best_cvar_history 사용**: 논문 그래프에는 누적 최고값
3. **이상치 처리**: iter 53-54 급락은 제외 또는 설명 필요
4. **현실적 기대**: 완벽한 robust optimization은 아니지만, 파라미터 최적화는 성공

---

**마지막 업데이트**: 2025-11-19 16:50
**실험 상태**: n_w=20으로 resume 중
**다음 확인**: 200 iterations 완료 후
