# 메모리/프로세스 셧다운 문제 분석 및 해결 (2025-11-13)

## 📊 문제 상황

### 실험 결과 패턴
| 시도 | 메모리 관리 방법 | 멈춘 지점 | 비고 |
|------|------------------|-----------|------|
| Trial 1 | 기본 (iteration 끝만) | **13번** | 첫 번째 벽 |
| Trial 2 | Iteration 끝 해제 강화 | **36번** | 2.7배 개선 ✨ |
| Trial 3 | GP 5번마다 + 과도한 해제 | **6번** | 오히려 악화 ❌ |
| Trial 4 | OpenCV 해제 추가 | **13번** | 첫 번째 벽 재발 |

### 공통 증상
- **에러 메시지 없이 조용히 종료** (exit code 없음)
- **13번 또는 36번에서 일관되게 멈춤**
- Windows Git Bash 환경

---

## 🔍 원인 분석 (Opus)

### 1. GPU 메모리 오버플로우 ⚠️
**문제점**:
- AirLine 모델들(DexiNed, OrientationDetector)이 GPU에 상주
- 매 이미지마다 GPU 연산 누적
- `torch.cuda.empty_cache()` 호출 부족

**근거**:
- Trial 2에서 iteration 끝 메모리 해제 강화 → 36번까지 개선
- GPU 메모리가 쌓이다가 임계점에서 프로세스 종료

### 2. 메모리 누수 패턴 ⚠️
**문제점**:
- `AirLine_assemble_test.py`에서 **전역 버퍼 사용** (TMP1, TMP2, TMP3)
- 이미지 증강 시 **float32 사용**으로 메모리 과다
- 119장 이미지를 **전체 메모리에 유지**

**근거**:
- OpenCV 메모리 해제 추가했는데도 13번에서 멈춤
- AirLine 내부의 전역 변수가 계속 누적

### 3. C++ 모듈 문제 (가능성)
**문제점**:
- `CRG311.pyd` C++ 확장 모듈의 메모리 관리 불명확
- 세그멘테이션 폴트 가능성

### 4. 리소스 제한
**문제점**:
- 파일 디스크립터 한계 도달 가능성
- 스레드 수 과다 생성

---

## ✅ 제안된 해결책

### 우선순위 1: GPU 메모리 관리 강화 (즉시 적용 가능)

#### A. GPU 메모리 80% 제한 설정
```python
import torch
import os

# optimization.py 시작 부분에 추가
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

#### B. 주기적 GPU 캐시 정리
```python
# BO iteration 루프에서
for iteration in range(n_iterations):
    # ... 기존 코드 ...

    # 매 iteration 끝
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # GPU 연산 완료 대기
    import gc
    gc.collect()

    # 5번마다 더 강력한 정리
    if (iteration + 1) % 5 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        gc.collect()  # 순환 참조 정리
```

---

### 우선순위 2: 체크포인트 시스템 (중요!)

**목표**: 프로세스가 터져도 중간 결과 보존 및 재시작 가능

#### 체크포인트 저장
```python
import json
from pathlib import Path

def save_checkpoint(iteration, train_X_full, train_Y, best_cvar_history,
                    best_params, checkpoint_dir):
    """5번마다 체크포인트 저장"""
    checkpoint = {
        'iteration': iteration,
        'train_X_full': train_X_full.cpu().numpy().tolist(),
        'train_Y': train_Y.cpu().numpy().tolist(),
        'best_cvar_history': best_cvar_history,
        'best_params': best_params,
    }

    checkpoint_file = checkpoint_dir / f"checkpoint_iter_{iteration:03d}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"  [Checkpoint] Saved at iteration {iteration}")

# BO 루프에서
for iteration in range(n_iterations):
    # ... 기존 코드 ...

    # 5번마다 체크포인트 저장
    if (iteration + 1) % 5 == 0:
        save_checkpoint(iteration + 1, train_X_full, train_Y,
                       best_cvar_history, best_params, checkpoint_dir)
```

#### 체크포인트에서 재시작
```python
def load_checkpoint(checkpoint_dir):
    """최신 체크포인트 로드"""
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_iter_*.json"))
    if not checkpoint_files:
        return None

    latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        checkpoint = json.load(f)

    print(f"  [Checkpoint] Loaded from {latest.name}")
    return checkpoint
```

---

### 우선순위 3: AirLine 메모리 관리

#### full_pipeline.py 개선
```python
def detect_with_full_pipeline(image, params, yolo_detector, ransac_weights=None):
    # ... 기존 코드 ...

    # ROI 루프 내부
    for cls, x1_roi, y1_roi, x2_roi, y2_roi in rois:
        # ... 처리 ...

        # ✅ 각 ROI 처리 후 즉시 메모리 해제
        del roi_bgr, roi_gray, roi_gray_blur, S_roi
        del roi_bgr_enhanced, roi_gray_enhanced, lines_by_algo

    # ✅ 함수 종료 전 최종 정리
    del processed_results, rois

    # ✅ GPU 사용했다면 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()

    return coords
```

---

### 우선순위 4: 진단 도구 (선택사항)

#### 간단한 메모리 모니터링
```python
import psutil
import torch

def log_memory_usage(iteration):
    """메모리 사용량 로깅"""
    process = psutil.Process()
    mem_info = process.memory_info()

    print(f"  [Memory] Iter {iteration}: "
          f"RAM={mem_info.rss / 1024**2:.1f}MB, "
          f"GPU={torch.cuda.memory_allocated() / 1024**2:.1f}MB")

# BO 루프에서 사용
for iteration in range(n_iterations):
    # ... 기존 코드 ...

    if (iteration + 1) % 5 == 0:
        log_memory_usage(iteration + 1)
```

---

## 🎯 적용 계획 (단계별)

### Phase 1: 즉시 적용 (가장 중요!)
1. ✅ **GPU 메모리 80% 제한** - 시작 부분에 추가
2. ✅ **주기적 GPU 캐시 정리** - iteration 끝과 5번마다
3. ✅ **체크포인트 시스템** - 5번마다 저장

**예상 효과**: 13번 → 30번 이상 기대

### Phase 2: 추가 개선 (시간 있으면)
4. ⏳ **메모리 모니터링 로그** - 문제 지점 파악
5. ⏳ **AirLine 메모리 관리 강화** - 전역 버퍼 정리

### Phase 3: 최후의 수단
6. ⚠️ **n_w 줄이기**: 3 → 2 (GP 차원 감소)
7. ⚠️ **이미지 수 줄이기**: 119장 → 50장
8. ⚠️ **배치 처리**: 이미지를 나눠서 처리

---

## 📝 Trial 2 성공 요인 분석

**Trial 2에서 36번까지 간 이유**:
```python
# optimization.py (Trial 2 코드)
# 5.11: 메모리 명시적 해제 (13번 iteration 문제 해결)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
import gc
gc.collect()
```

**핵심**:
- **간단하지만 일관된 메모리 해제**
- 과도한 `del` 명령 없이 기본만 충실히

**교훈**:
- 복잡한 메모리 관리보다 **기본에 충실**
- **GPU 캐시 정리**가 가장 중요
- `del` 명령 남발은 오히려 불안정

---

## 🚀 다음 실험 전략

### 전략 A: Trial 2 코드 기반 + GPU 강화
```python
# Trial 2의 간단한 메모리 해제
+ GPU 메모리 80% 제한
+ GPU synchronize 추가
+ 체크포인트 5번마다
```

**목표**: 36번 → 50번

### 전략 B: 더 보수적 접근
```python
# 위 전략 A
+ n_w = 3 → 2
+ iterations = 50 → 30 (더 짧게)
```

**목표**: 일단 완주 보장

---

## 💡 핵심 인사이트

1. **GPU 메모리가 주범**: AirLine 모델들이 GPU에 상주하며 누적
2. **간단함이 최고**: 복잡한 메모리 관리는 오히려 불안정
3. **체크포인트 필수**: 언제 터질지 모르니 중간 저장 필수
4. **13번과 36번**: 일관된 패턴 = 메모리 한계점

---

**작성일**: 2025-11-13
**상태**: Phase 1 적용 대기 중
**다음**: GPU 메모리 관리 + 체크포인트 적용 → 재실험
