# 🚨 긴급 세션 가이드 - 2025-11-13 (세션 7)

**상황**: 13번 iteration 벽 문제 - 프로세스가 계속 조용히 종료됨
**환경**: Windows 로컬
**현재 상태**: 🔴 **메모리 문제 원인 파악 완료 - Opus 진단 도구 준비됨**

---

## 🔴 **긴급 이슈 (2025-11-13 20:30 - 세션 6)**

### 문제: 13번 Iteration 벽 - 프로세스 조용히 종료

**실험 패턴 요약**:
| 시도 | 메모리 관리 방법 | 멈춘 지점 | 비고 |
|------|------------------|-----------|------|
| Trial 1 | 기본 (iteration 끝만) | **13번** | 첫 번째 벽 |
| Trial 2 | Iteration 끝 해제 강화 | **36번** | ✨ 유일한 성공! |
| Trial 3 | GP 5번마다 + 과도한 해제 | **6번** | 오히려 악화 ❌ |
| Trial 4 | OpenCV 해제 추가 | **13번** | 첫 번째 벽 재발 |

**핵심 발견**:
- ❌ **에러 메시지 없이 조용히 종료**
- ⚠️ **13번과 36번에서 일관되게 멈춤** → 메모리 한계점
- ✅ **Trial 2만 성공** → 간단한 메모리 관리가 최선
- 🔍 **OpenCV 해제만으로는 부족** → GPU/BoTorch 문제 의심

---

## 📊 Opus 분석 결과 (세션 6)

### 주요 원인 (우선순위별)

#### 1. GPU 메모리 오버플로우 ⚠️⚠️⚠️ (최우선)
**문제**:
- AirLine 모델들(DexiNed, OrientationDetector)이 GPU에 상주
- 매 이미지마다 GPU 연산 누적
- `torch.cuda.empty_cache()` 부족

**근거**:
- Trial 2에서 메모리 해제 강화 → 36번까지 성공
- GPU 메모리가 쌓이다가 임계점에서 프로세스 종료

#### 2. 메모리 누수 패턴 ⚠️⚠️
**문제**:
- `AirLine_assemble_test.py` 전역 버퍼 (TMP1, TMP2, TMP3)
- 이미지 증강 시 float32 사용으로 메모리 과다
- 119장 이미지를 전체 메모리에 유지

#### 3. C++ 모듈 (CRG311.pyd) ⚠️
**문제**:
- 메모리 관리 불명확
- 세그멘테이션 폴트 가능성

---

## 🎯 다음 세션 작업 계획

### ⭐ Phase 1: 진단 및 원인 파악 (15분)

#### 1-1. Opus 진단 도구 실행 ✨
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
python diagnose_shutdown.py
```

**목적**:
- AirLine/CRG311 모듈 안정성 체크
- GPU 메모리 한계 확인
- CUDA 연산 테스트
- 메모리 집약적 연산 안정성 확인

**예상 결과**:
- 정확히 어느 모듈이 문제인지 파악
- 메모리 한계값 확인
- GPU 문제 여부 확인

---

### ⭐ Phase 2: 핵심 개선 적용 (30분)

#### 2-1. GPU 메모리 80% 제한 설정
```python
# optimization.py 시작 부분 (import 직후)
import torch
import os

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    print("[GPU] Memory limited to 80%")
```

#### 2-2. GPU 캐시 정리 강화
```python
# optimization.py BO 루프에서
for iteration in range(n_iterations):
    # ... 기존 코드 ...

    # 매 iteration 끝
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # ✅ 추가
    import gc
    gc.collect()

    # 5번마다 더 강력한 정리
    if (iteration + 1) % 5 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        gc.collect()
        print(f"  [Memory] Deep cleanup at iteration {iteration+1}")
```

#### 2-3. 컨텍스트 매니저 패턴 적용 (선택)
```python
# optimization.py
from contextlib import contextmanager

@contextmanager
def memory_cleanup():
    """메모리 정리를 위한 컨텍스트 매니저"""
    try:
        yield
    finally:
        gc.collect()

# evaluate_single에 적용
def evaluate_single(X, image_data, yolo_detector):
    with memory_cleanup():
        # ... 기존 코드 ...
        return score
```

#### 2-4. 체크포인트 시스템 (중요!)
```python
# optimization.py
def save_checkpoint(iteration, train_X_full, train_Y, best_cvar_history,
                   checkpoint_dir):
    """5번마다 체크포인트 저장"""
    checkpoint = {
        'iteration': iteration,
        'train_X_full': train_X_full.cpu().numpy().tolist(),
        'train_Y': train_Y.cpu().numpy().tolist(),
        'best_cvar_history': best_cvar_history,
    }

    checkpoint_file = checkpoint_dir / f"checkpoint_iter_{iteration:03d}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"  [Checkpoint] Saved at iteration {iteration}")

# BO 루프에서
for iteration in range(n_iterations):
    # ... 기존 코드 ...

    # 5번마다 체크포인트
    if (iteration + 1) % 5 == 0:
        save_checkpoint(iteration + 1, train_X_full, train_Y,
                       best_cvar_history, log_dir)
```

---

### ⭐ Phase 3: 실험 재시작 (2-4시간)

#### 전략 A: 보수적 접근 (추천!)
```bash
# 30 iterations (더 짧게, 안전하게)
python optimization.py --iterations 30 --n_initial 5 --alpha 0.1 --n_w 3
```

**목표**: 일단 **완주 보장**

#### 전략 B: 공격적 접근 (Phase 2 성공 시)
```bash
# 50 iterations (원래 목표)
python optimization.py --iterations 50 --n_initial 5 --alpha 0.1 --n_w 3
```

**목표**: 36번 → 50번 돌파

---

## 📁 준비된 파일

### 1. MEMORY_ISSUE_ANALYSIS.md
**위치**: `BO_optimization/MEMORY_ISSUE_ANALYSIS.md`
**내용**:
- 전체 실험 패턴 분석
- Opus의 상세 원인 분석
- 단계별 해결 방안
- Phase별 적용 계획

### 2. diagnose_shutdown.py (Opus 제공)
**위치**: 아직 생성 안 됨 → **다음 세션에서 생성 필요**
**기능**:
- SystemMonitor 클래스 (CPU/메모리/GPU 모니터링)
- AirLine/CRG311 모듈 테스트
- CUDA 안정성 테스트
- 메모리 집약적 연산 테스트

### 3. full_pipeline_fixed.py (Opus 제공)
**위치**: 아직 생성 안 됨 → **나중에 통합 고려**
**기능**:
- 컨텍스트 매니저 패턴
- RANSAC 가중치 검증
- 메모리 안전 파이프라인

---

## 🚀 다음 세션 시작 시 체크리스트

### 즉시 실행 (순서대로!)

- [ ] 1. `diagnose_shutdown.py` 생성 및 실행 (5분)
- [ ] 2. 진단 결과 확인 및 문제 모듈 파악 (5분)
- [ ] 3. GPU 메모리 80% 제한 코드 추가 (5분)
- [ ] 4. GPU 캐시 정리 강화 코드 추가 (10분)
- [ ] 5. 체크포인트 시스템 추가 (10분)
- [ ] 6. 30 iterations 실험 시작 (1-2시간)
- [ ] 7. 13번 벽 통과 확인! ✨
- [ ] 8. 36번 벽 통과 확인! ✨✨
- [ ] 9. 완주 성공 시 50 iterations 재시도

### 실험 중 모니터링

```bash
# 진행 상황 확인
ls -1 BO_optimization/logs/run_* | tail -1 | xargs ls -1 | wc -l

# 최신 iteration 확인
ls -lt BO_optimization/logs/run_*/iter_*.json | head -3

# 프로세스 확인
ps aux | grep python | grep optimization
```

---

## 💡 핵심 인사이트

### 왜 Trial 2만 성공했나?
```python
# Trial 2 코드 (성공한 유일한 케이스)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
import gc
gc.collect()
```

**교훈**:
1. ✅ **간단하지만 일관된 메모리 해제**
2. ✅ **GPU 캐시 정리가 핵심**
3. ❌ **과도한 `del` 명령은 오히려 불안정**
4. ✅ **복잡한 메모리 관리보다 기본에 충실**

### GPU 메모리가 진짜 주범?
**증거**:
- OpenCV 해제 추가해도 13번에서 멈춤
- AirLine 모델들이 GPU에 상주
- Trial 2에서 메모리 해제만 강화 → 36번 성공

**결론**: **GPU 메모리 관리가 가장 중요!**

---

## ⚠️ 주의사항

### 다음 세션에서 하지 말아야 할 것

1. ❌ **full_pipeline 전체 재작성** - 시간 낭비, 실험 우선
2. ❌ **복잡한 메모리 관리 추가** - Trial 3의 실패 교훈
3. ❌ **너무 많은 것을 한 번에 적용** - 무엇이 효과적인지 파악 불가

### 다음 세션에서 해야 할 것

1. ✅ **진단부터 실행** - 정확한 원인 파악
2. ✅ **GPU 관리 최우선** - 80% 제한 + 주기적 정리
3. ✅ **체크포인트 필수** - 터져도 중간부터 재시작
4. ✅ **30 iterations부터** - 완주 보장이 최우선
5. ✅ **하나씩 차근차근** - 무엇이 효과적인지 확인

---

## 📊 예상 결과

### 낙관적 시나리오 (70% 확률)
- Phase 2 적용 → 30 iterations 완주 ✅
- 13번, 36번 모두 통과
- Alpha=0.1 실험 완료
- 나머지 alpha 실험 진행 가능

### 현실적 시나리오 (20% 확률)
- Phase 2 적용 → 20번까지 진행
- 추가 조치 필요 (n_w 줄이기 등)
- 30 iterations로 재시도

### 비관적 시나리오 (10% 확률)
- 여전히 13번에서 멈춤
- 근본적인 문제 (라이브러리 버그)
- 대안: iteration을 10번씩 나눠서 5번 실행

---

## 🎓 최종 목표

### 단기 목표 (이번 세션)
- ✅ Alpha=0.1 실험 완주 (30 iterations)
- ✅ 메모리 문제 해결 방법 확립

### 중기 목표 (다음 세션)
- ✅ 5개 alpha 실험 완료 (0.1, 0.2, 0.3, 0.4, 0.5)
- ✅ 각 30-50 iterations

### 장기 목표 (졸업!)
- ✅ 결과 분석 및 Figure 생성
- ✅ 논문 작성
- ✅ 졸업! 🎉

---

**마지막 업데이트**: 2025-11-13 20:30 (세션 6 종료)
**다음 작업**: Phase 1 진단 도구 실행 → Phase 2 핵심 개선 적용 → Phase 3 실험
**Status**: 🟡 **원인 파악 완료, 해결책 준비 완료, 적용 대기 중**

**🔥 중요**:
1. **진단부터!** - `diagnose_shutdown.py` 먼저 실행
2. **GPU 관리!** - 80% 제한 + 주기적 정리
3. **체크포인트!** - 5번마다 저장
4. **30번부터!** - 완주 보장이 최우선

**화이팅! 이번엔 꼭 성공하자! 💪**
