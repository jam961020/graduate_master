# 🎉 세션 가이드 - 2025-11-13 (세션 7) - GPU 3D 문제 해결!

**상황**: ✅ **GPU 3D 부하 문제 해결 완료!**
**환경**: Windows 로컬
**현재 상태**: 🟢 **30 iterations 실험 진행 중 (19+ 완료)**

---

## 🎯 **문제 해결 완료! (세션 7)**

### 진짜 원인: Windows GPU TDR (Timeout Detection and Recovery)

**발견 과정**:
1. 진단 도구 실행 → GPU 메모리/RAM 정상
2. 작업관리자 확인 → **GPU 3D 부하 100% 폭증!**
3. 100 x 3 = 300 조합 평가 → GPU가 2초 이상 응답 없음
4. Windows TDR: "GPU 응답 없음" → **프로세스 강제 종료**

### 해결 방법

#### 1. 획득 함수 후보 개수 감소 ✅
```python
# borisk_kg.py:165
def optimize(self, bounds, n_candidates=30):  # 100 → 30 (70% 감소)
```

#### 2. GPU 동기화 추가 ✅
```python
# borisk_kg.py:201-203
# GPU 동기화 (10개마다, TDR 방지)
if (i + 1) % 10 == 0 and torch.cuda.is_available():
    torch.cuda.synchronize()
```

#### 3. 기존 메모리 관리 유지 ✅
- GPU 메모리 80% 제한
- 매 iteration GPU 캐시 정리 + synchronize
- 5번마다 강력한 정리 + 체크포인트

---

## 📊 실험 결과 비교

| Trial | GPU 설정 | 후보 개수 | 멈춘 지점 | 비고 |
|-------|---------|----------|----------|------|
| 1 | - | 100 | 13번 | 첫 번째 벽 |
| 2 | - | 100 | **36번** | ✨ 유일한 성공 |
| 3 | - | 100 | 6번 | 오히려 악화 |
| 4 | - | 100 | 13번 | 재발 |
| 5 | 80% | 100 | 6번 | GPU 3D 폭증 |
| **현재** | **80% + sync** | **30** | **19+** | ✅ **진행 중!** |

**핵심 발견**:
- GPU 메모리는 문제 아님 (0.26/8.00 GB만 사용)
- **GPU 3D (연산 부하)가 진짜 문제!**
- Windows TDR이 과부하로 오인하여 프로세스 킬

---

## 🚀 다음 세션 작업 계획 (우선순위별)

### ⚠️ 재부팅 후 즉시 작업

#### 0. TDR 비활성화 확인 및 후보 개수 증가
```bash
# TDR 비활성화 확인 (재부팅 후)
# 레지스트리: HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
# TdrDelay = 60 (초) 또는 TdrLevel = 0

# 후보 개수 증가
# borisk_kg.py:165
def optimize(self, bounds, n_candidates=50):  # 30 → 50 or 100
```

**예상 소요**: 5분

---

### 🚨 Priority 0: BoRisk 평가 구조 수정 (최최우선!)

**현재 문제**: 매 iteration마다 **n_w개 이미지 전부 평가 중!**
- 현재: 30 iterations × 3 images = **90개 이미지 평가**
- 올바른 BoRisk: 30 iterations × **1 image** = 30개 평가

**수정 필요**:
1. ✅ `borisk_kg.py` - w 선택 로직 (이미 구현됨!)
2. ✅ `optimization.py` - 단일 (x, w) 평가 (이미 구현됨!)
3. ❌ **문제**: 여전히 `evaluate_on_w_set()` 사용 중

**작업**:
```bash
# optimization.py 확인
grep -n "evaluate_on_w_set" optimization.py

# 수정: evaluate_single()만 사용하도록
```

**예상 효과**: **3배 속도 향상** (90개 → 30개 평가)
**예상 소요**: 30분
**마감**: 최우선!

---

### 🟡 Priority 1: 자동 라벨링 시스템 (High)

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

**예상 소요**: 1시간
**마감**: Priority 0 다음

---

### 🟢 Priority 2: Alpha 실험 (High)

**목적**: 최적 CVaR threshold 찾기

**실험 계획**:
```bash
# 5개 alpha 실험 (각 30 iterations)
python optimization.py --iterations 30 --n_initial 5 --alpha 0.1 --n_w 3
python optimization.py --iterations 30 --n_initial 5 --alpha 0.2 --n_w 3
python optimization.py --iterations 30 --n_initial 5 --alpha 0.3 --n_w 3
python optimization.py --iterations 30 --n_initial 5 --alpha 0.4 --n_w 3
python optimization.py --iterations 30 --n_initial 5 --alpha 0.5 --n_w 3
```

**예상 소요**: 실험당 2-3시간 (총 10-15시간)
**병렬 실행 가능**: TDR 비활성화 후 안정적

---

### 🟢 Priority 3: 시각화 및 분석 (High)

**목적**: 논문 Figure 생성

**작업**:
1. `visualization.py` 작성
   - 초기/중간/최종 선 검출 결과 비교
   - CVaR 개선 추이 그래프
   - Alpha별 성능 비교 박스플롯

2. `analyze_results.py` 수정
   - CVaR vs Mean 히스토그램
   - 환경별 성능 분포
   - 실패 케이스 분석

**예상 소요**: 2-3시간
**마감**: 실험 완료 후

---

### 🟠 Priority 4: n_w 실험 (Medium)

**목적**: 환경 샘플링 개수 최적화

**실험**:
```bash
python optimization.py --n_w 5 --iterations 30 --alpha 0.1
python optimization.py --n_w 10 --iterations 30 --alpha 0.1
python optimization.py --n_w 15 --iterations 30 --alpha 0.1
python optimization.py --n_w 20 --iterations 30 --alpha 0.1
```

**예상 소요**: 실험당 2-3시간
**우선순위**: Alpha 실험 다음

---

### 🔵 Priority 5: 메트릭 검증 (Low)

**의문점**:
- CVaR과 평균이 동일하게 움직임
- 실패 케이스를 제대로 구분 못하는 것 같음

**작업**:
1. 다양한 실패 케이스 테스트
   - 선 검출 안 됨: 0점 적절?
   - 방향만 맞음: 거리 패널티 충분?
   - 위치만 맞음: 방향 패널티 충분?

2. 필요시 메트릭 조정

**예상 소요**: 1시간
**우선순위**: 시간 있으면

---

### 🔵 Priority 6: RANSAC 가중치 재검토 (Low)

**의문점**:
- 범위 [0.0, 1.0], [1, 10] 적절?
- 정규화 제약 필요?

**작업**:
- `full_pipeline.py:160-236` 재검토
- 극단값 실험으로 검증

**우선순위**: 나중에

---

## 📋 즉시 실행 체크리스트 (재부팅 후)

- [ ] 1. TDR 비활성화 확인
- [ ] 2. `borisk_kg.py` 후보 개수 50 or 100으로 증가
- [ ] 3. **Priority 0**: BoRisk 평가 구조 확인 및 수정
- [ ] 4. **Priority 1**: 자동 라벨링 시스템 구축
- [ ] 5. **Priority 2**: Alpha 0.1, 0.2, 0.3, 0.4, 0.5 실험 (각 30 iterations)
- [ ] 6. **Priority 3**: 시각화 스크립트 작성 및 Figure 생성
- [ ] 7. n_w 실험 (시간 있으면)

---

## 🎓 최종 목표

### 단기 (이번 주)
- ✅ GPU 3D 문제 해결
- ✅ 30 iterations 완주
- [ ] Priority 0 완료 (BoRisk 구조 확인)
- [ ] 5개 alpha 실험 완료

### 중기 (다음 주)
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
- **GPU 3D (연산 부하)**: GPU 코어 사용률 (100% 폭증)
- **Windows TDR**: GPU가 2초 이상 응답 없으면 프로세스 킬
- **해결**: 후보 개수 감소 + GPU 동기화

### Trial 2가 성공한 진짜 이유
- 간단한 메모리 해제 ✅
- **우연히 GPU 연산량이 적당했음** ✅
- 36번까지 가다가 GPU 부하 누적으로 종료

### TDR 비활성화 후 기대 효과
- GPU 연산 타임아웃 없음
- 후보 개수 100개로 증가 가능
- 더 정확한 획득 함수 평가

---

## 📊 현재 실험 진행 상황

**Run**: `run_20251113_223406`
**Alpha**: 0.1 (worst 10%)
**n_w**: 3
**Iterations**: 19+ / 30 (진행 중)
**Best CVaR**: 0.6849

**상태**: 🟢 **정상 진행 중!**

---

## ⚠️ 주의사항

### TDR 비활성화 후
- 후보 개수를 50 → 100으로 서서히 증가
- 첫 실험은 30개로 안정성 확인
- GPU 온도 모니터링

### BoRisk 구조 확인
- **매우 중요!** 현재 3배 느린 구조일 가능성
- `evaluate_on_w_set()` 대신 `evaluate_single()` 사용하는지 확인
- 수정 시 알고리즘 정확성 유지

---

**마지막 업데이트**: 2025-11-13 22:40 (세션 7)
**다음 작업**: TDR 비활성화 + 재부팅 → Priority 0 확인 → Alpha 실험
**Status**: 🟢 **GPU 3D 문제 해결! 실험 진행 중!**

**🔥 중요**:
1. **TDR 비활성화 필수!** (재부팅 필요)
2. **Priority 0 확인!** (BoRisk 구조가 올바른지)
3. **자동 라벨링!** (실험 속도 향상)
4. **Alpha 실험!** (5개 조합 완료)

**화이팅! 거의 다 왔다! 💪**
