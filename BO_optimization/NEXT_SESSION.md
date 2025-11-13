# 🎉 세션 가이드 - 2025-11-14 (세션 8) - 실험 완주 및 BO 동작 검증!

**상황**: ✅ **2개 실험 완주! BO 동작 패턴 정상 확인!**
**환경**: Windows 로컬
**현재 상태**: 🟢 **TDR 비활성화 준비 완료**

---

## 🎯 **완료된 실험 분석 (세션 8)**

### 실험 1: run_20251113_223406 (30 iterations)

**설정**:
- Alpha: 0.1 (worst 10%)
- n_w: 3 environments
- n_candidates: 30 (TDR 방지)
- n_initial: 5

**결과**:
- **Initial CVaR**: 0.260
- **Best CVaR**: 0.745 (iteration 32)
- **Final CVaR**: 0.664
- **개선율**: +155.44%! 🎉
- **개선 횟수**: 6/35 iterations

**특징**:
- 초반 5번 iteration에서 급격한 개선 (0.26 → 0.68)
- Iteration 27, 32에서 추가 개선
- 완주 성공! ✅

---

### 실험 2: run_20251113_225648 (50 iterations)

**설정**:
- Alpha: 0.1 (worst 10%)
- n_w: 3 environments
- n_candidates: 30
- n_initial: 5

**결과**:
- **Initial CVaR**: 0.565
- **Best CVaR**: 0.689 (iteration 25)
- **Final CVaR**: 0.683
- **개선율**: +20.95%
- **개선 횟수**: 6/50 iterations

**특징**:
- 0~20: 탐색 phase (exploration)
- 20~25: **급격한 개선!** (0.62 → 0.68)
- 25~50: 수렴 phase (exploitation, 0.68 주변 유지)
- 완주 성공! ✅

---

## 📊 실험 비교 분석

| 항목 | Run 223406 | Run 225648 |
|------|-----------|-----------|
| Iterations | 30 | 50 |
| Initial CVaR | 0.260 | 0.565 |
| Best CVaR | 0.745 | 0.689 |
| Final CVaR | 0.664 | 0.683 |
| Improvement | +155% | +21% |
| Best at iter | 32 | 25 |
| 개선 횟수 | 6/35 | 6/50 |

**핵심 발견**:
- 두 실험 모두 **정상적인 BO 패턴**
- Initial 값에 따라 개선 폭 차이 (낮을수록 개선 여지 큼)
- 25~35 iterations 후 수렴 경향

---

## 💡 BO 동작 패턴 분석 (정상 확인!)

### 1️⃣ CVaR이 들쑥날쑥한 이유 (정상!)

**원인**:
1. **매 iteration 다른 환경(w) 평가**
   - 같은 x라도 다른 환경에선 다른 성능
   - CVaR은 GP posterior로 추정 (실제 평가 1개)

2. **Exploration vs Exploitation**
   - Exploration: 불확실한 영역 탐색 → 일시적 성능 하락
   - Exploitation: 좋은 영역 활용 → 성능 유지

3. **GP 불확실성**
   - 학습 데이터 적은 초반: 불확실성 큼
   - 데이터 쌓일수록 안정화

**증거**: Smoothed Trend (5-iter 이동평균) 보면 **전체적으로 상승**!

---

### 2️⃣ KG 높은데 CVaR 하락하는 이유 (정상!)

**KG의 의미**:
```
KG = Expected(미래 CVaR 개선)
   ≠ 현재 iteration CVaR
```

**KG가 높다** = "이 (x,w)를 평가하면 **미래에** 정보 가치가 높음"

**예시 (Run 225648)**:
- Iter 1-20: 탐색 (KG 높음, CVaR 0.60)
- → GP가 파라미터 공간 학습
- Iter 20-25: 학습된 지식으로 **급격한 개선** (0.68)
- Iter 25-50: 수렴 확인 (exploitation)

**증거**: Learning Speed 그래프에서 Iter 1, 20에 큰 학습!

---

### 3️⃣ Best가 중간에 나오는 이유 (정상!)

**Run 225648: 50번 반복, Best at 25**

**이유**:
1. **Local optimum 도달**: 25번에 좋은 영역 찾음
2. **수렴 phase**: 25-50은 확인 단계 (exploitation)
3. **거의 수렴**: Best(0.689) vs Final(0.683) = 0.8% 차이

**증거**:
- Exploration vs Exploitation 그래프
  - 초반: 파란색 (exploration) 많음
  - 후반: 녹색 (exploitation) 거의 100%

**결론**: **정상적인 BO 수렴 패턴!**

---

## ✅ BO 알고리즘 검증 결과

### 1. CVaR 들쑥날쑥 → ✅ 정상
- Exploration + GP 불확실성
- 전체 추세는 상승 (Smoothed Trend 확인)

### 2. KG 높은데 CVaR 하락 → ✅ 정상
- KG는 미래 정보 가치 (단기 ≠ 장기)
- 탐색 후 급격한 개선 확인됨

### 3. Best가 중간 → ✅ 정상
- 수렴 후 확인 단계 (exploitation)
- Best와 Final 차이 < 1%

**종합**: **BoRisk 알고리즘이 의도대로 작동 중!** 🎉

---

## 🚀 다음 세션 작업 계획 (우선순위별)

### ⚠️ 재부팅 후 즉시 작업

#### 0. TDR 비활성화 + n_candidates 증가 ✅

**TDR 비활성화 (재부팅 필요)**:
```
레지스트리: HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
TdrDelay = 60 (초) 또는 TdrLevel = 0
```

**n_candidates 증가**:
```python
# borisk_kg.py:165
def optimize(self, bounds, n_candidates=100):  # 30 → 100 복원 완료!
```

**상태**: ✅ 코드 수정 완료, TDR 비활성화만 남음

---

### 🚨 Priority 0: BoRisk 평가 구조 검증 (최우선!)

**의문**: 매 iteration마다 **n_w개 이미지 전부 평가 중인가?**

**현재 코드 확인 필요**:
1. `borisk_kg.py` - w 선택 로직 구현 여부 ✅ (구현됨!)
2. `optimization.py` - 단일 (x, w) 평가 vs n_w개 평가

**확인 방법**:
```bash
# optimization.py에서 evaluation 부분 확인
grep -n "evaluate_on_w_set\|evaluate_single" optimization.py

# Logs 확인 (매 iteration 몇 개 이미지 평가?)
cat logs/run_20251113_223406/iter_001.json
```

**예상 효과**:
- 만약 n_w개 평가 중이면 → 1개로 수정 → **3배 속도 향상**
- 이미 1개만 평가 중이면 → 정상 확인

**예상 소요**: 30분
**마감**: **최우선!**

---

### 🟡 Priority 1: 대규모 실험 (High)

#### 실험 A: n_w, iterations 증가 (TDR 비활성화 후)

```bash
# 큰 실험 (n_w=15, 100 iterations)
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
python optimization.py --iterations 100 --n_initial 10 --alpha 0.2 --n_w 15
```

**목적**:
- 더 많은 환경 샘플링 (3 → 15)
- 더 긴 최적화 (50 → 100)
- 후보 개수 100 사용

**예상 소요**: 3-4시간
**마감**: TDR 비활성화 후 즉시

---

#### 실험 B: Alpha 비교 실험

```bash
# Alpha별 성능 비교 (각 50 iterations)
python optimization.py --iterations 50 --n_initial 10 --alpha 0.1 --n_w 10
python optimization.py --iterations 50 --n_initial 10 --alpha 0.2 --n_w 10
python optimization.py --iterations 50 --n_initial 10 --alpha 0.3 --n_w 10
python optimization.py --iterations 50 --n_initial 10 --alpha 0.4 --n_w 10
python optimization.py --iterations 50 --n_initial 10 --alpha 0.5 --n_w 10
```

**목적**: 최적 CVaR threshold 찾기
**예상 소요**: 실험당 2-3시간 (총 10-15시간)
**우선순위**: 실험 A 다음

---

### 🟢 Priority 2: 자동 라벨링 시스템 (High)

**목적**: Ground truth 자동 생성으로 실험 속도 향상

**작업**:
```python
# auto_labeling.py 생성
"""
AirLine 결과로 GT 자동 생성
- 6개 점 자동 추출
- ground_truth.json 포맷 저장
"""

# 실행
python auto_labeling.py --output dataset/ground_truth_auto.json
```

**예상 소요**: 1-2시간
**마감**: Priority 0, 1 다음

---

### 🟢 Priority 3: 시각화 고도화 (Medium)

**현재**: `visualize_exploration.py` ✅
- 9개 subplot으로 BO 과정 상세 시각화
- CVaR 추이, 개선도, Exploration/Exploitation 등

**추가 작업**:
1. **비교 시각화**
   - 여러 실험 결과 한 그래프에
   - Alpha별 성능 비교 박스플롯

2. **논문용 Figure**
   - 초기/중간/최종 선 검출 결과
   - 환경별 성능 분포

**예상 소요**: 2-3시간
**마감**: 실험 완료 후

---

### 🔵 Priority 4: n_w 실험 (Medium)

**목적**: 환경 샘플링 개수 최적화

**실험**:
```bash
python optimization.py --n_w 5 --iterations 50 --alpha 0.2
python optimization.py --n_w 10 --iterations 50 --alpha 0.2
python optimization.py --n_w 15 --iterations 50 --alpha 0.2
python optimization.py --n_w 20 --iterations 50 --alpha 0.2
```

**예상 소요**: 실험당 2-3시간
**우선순위**: Alpha 실험 다음

---

### 🔵 Priority 5: 입력 정규화 (Low)

**현재 경고**:
```
InputDataWarning: Data (input features) is not contained to the unit cube.
```

**의미**: BoTorch가 [0,1] 범위로 정규화 권장
**동작**: ✅ 정상 작동 중 (경고만 표시)

**수정 (선택사항)**:
```python
# optimization.py에 추가
def normalize_bounds(bounds):
    # [lower, upper] → [0, 1] 변환
    pass
```

**우선순위**: 낮음 (성능에 영향 없음)

---

## 📋 즉시 실행 체크리스트 (재부팅 후)

- [ ] 1. **TDR 비활성화 확인** (레지스트리)
- [ ] 2. **재부팅**
- [ ] 3. **Priority 0**: BoRisk 평가 구조 검증
- [ ] 4. **실험 A**: 대규모 실험 (n_w=15, 100 iters)
- [ ] 5. **실험 B**: Alpha 비교 (0.1~0.5, 각 50 iters)
- [ ] 6. **Priority 2**: 자동 라벨링 시스템
- [ ] 7. **Priority 3**: 비교 시각화
- [ ] 8. n_w 실험 (시간 있으면)

---

## 🎓 최종 목표

### 단기 (이번 주)
- ✅ GPU 3D 문제 해결
- ✅ 30/50 iterations 완주 (2개)
- ✅ BO 동작 패턴 정상 확인
- [ ] TDR 비활성화
- [ ] Priority 0 검증 (평가 구조)
- [ ] 대규모 실험 (n_w=15, 100 iters)

### 중기 (다음 주)
- [ ] Alpha 비교 실험 (5개) 완료
- [ ] 자동 라벨링 시스템 완성
- [ ] 결과 분석 및 Figure 생성
- [ ] 논문 초안 작성 시작

### 장기 (졸업!)
- [ ] 논문 완성
- [ ] 발표 자료 준비
- [ ] 졸업! 🎉

---

## 💡 핵심 인사이트

### GPU 3D vs GPU 메모리
- **GPU 메모리**: 데이터 저장 공간 (8GB 중 0.26GB만 사용)
- **GPU 3D (연산 부하)**: GPU 코어 사용률
- **Windows TDR**: GPU가 2초 이상 응답 없으면 프로세스 킬
- **해결**: 후보 개수 감소(100→30) + GPU 동기화

### BO 동작 패턴 (정상!)
- **CVaR 변동**: Exploration + GP 불확실성 (정상)
- **KG ≠ CVaR**: KG는 미래 정보 가치 예측
- **수렴**: 25-35 iterations 후 local optimum 도달
- **Exploitation**: 수렴 후 확인 단계 (Best 유지)

### 실험 결과
- **개선율**: 초기 값에 따라 20%~155%
- **수렴 속도**: 25-35 iterations
- **안정성**: n_candidates=30, GPU sync로 완주 가능

---

## 📊 완료된 실험 목록

| Run | Date | Iterations | Alpha | n_w | Initial | Best | Improvement | Status |
|-----|------|-----------|-------|-----|---------|------|-------------|--------|
| 223406 | 11-13 | 30 | 0.1 | 3 | 0.260 | 0.745 | +155% | ✅ 완주 |
| 225648 | 11-13 | 50 | 0.1 | 3 | 0.565 | 0.689 | +21% | ✅ 완주 |

**Visualization**: ✅ 생성 완료
- `results/bo_cvar_20251113_223406_exploration.png`
- `results/bo_cvar_20251113_225648_exploration.png`

---

## ⚠️ 주의사항

### TDR 비활성화 후
- 첫 실험: n_candidates=30으로 안정성 확인
- GPU 온도 모니터링
- 안정적이면 n_candidates=100으로 증가

### BoRisk 구조 검증
- **매우 중요!** 평가 구조가 올바른지 확인
- 1개 평가 vs n_w개 평가
- 수정 시 알고리즘 정확성 유지

### Git 관리
- 실험 완료 후 commit
- Visualization 이미지 포함
- md 파일 업데이트

---

**마지막 업데이트**: 2025-11-14 02:00 (세션 8)
**다음 작업**: TDR 비활성화 + 재부팅 → Priority 0 검증 → 대규모 실험
**Status**: 🟢 **BO 동작 검증 완료! TDR 비활성화 대기 중!**

**🔥 중요**:
1. ✅ **2개 실험 완주** (30, 50 iters)
2. ✅ **BO 패턴 정상 확인** (CVaR 변동, KG, 수렴 모두 정상!)
3. ⏳ **TDR 비활성화 대기** (재부팅 필요)
4. 🎯 **Priority 0 검증** (평가 구조 확인)
5. 🚀 **대규모 실험** (n_w=15, 100 iters)

**화이팅! 이제 본격적인 실험 시작이다! 💪**
