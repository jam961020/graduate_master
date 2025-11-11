# CRG311 Segfault 디버깅 리포트
**날짜**: 2025-11-11 21:40
**소요 시간**: 약 30분

---

## 문제 요약
**증상**: `optimization.py` 실행 시 segmentation fault 발생
**원인**: AirLine_assemble_test.py의 하드코딩된 Windows 경로

---

## 발견한 문제들

### 1. Windows 경로 하드코딩 ✅ 수정 완료
**위치**: AirLine_assemble_test.py:47
```python
sys.path.append(r"C:\Users\user\Desktop\study\task\weld2025\AirLine\build")
```
**수정**: 주석 처리
```python
# sys.path.append(r"C:\Users\user\Desktop\study\task\weld2025\AirLine\build")  # Windows only
```

### 2. MLP 모델 경로 하드코딩 ✅ 수정 완료
**위치**: AirLine_assemble_test.py:128
```python
MLP_MODEL_PATH = r"C:\Users\user\Desktop\...\model_A.pth"
```
**수정**: None으로 변경
```python
MLP_MODEL_PATH = None  # Disabled for Linux
```

### 3. 무거운 Import들 ✅ Try-Except 추가
**위치**: AirLine_assemble_test.py:33-36
- `from abs_6_dof import *`
- `from run_inference import ...`
- `from pendant_inference import ...`

**수정**: 모두 try-except로 감싸서 실패해도 계속 진행

---

## 테스트 결과

### ✅ 성공한 테스트
1. **CRG311 모듈 로딩**: 정상
2. **CRG311.desGrow() 직접 호출**: 정상 작동
3. **DexiNed 모델 로딩 및 추론**: 정상 작동
4. **AirLine_assemble_test import**: 정상

### ❌ 여전히 실패하는 테스트
1. **test_airline_final.py 실행**: Segfault (exit 139)
2. **optimization.py 실행**: Timeout 또는 segfault

---

## 추가 조사 필요사항

### 의심되는 원인
1. **실행 컨텍스트 차이**:
   - Python `-c` 옵션 실행: 성공
   - 파일 직접 실행: Segfault
   - 차이점: 작업 디렉토리, `__file__` 경로 등

2. **_init_airline_models() 함수**:
   - DexiNed 모델 로딩 시 GPU 메모리 할당 문제 가능성
   - 특정 조건에서만 crash 발생

3. **카메라 파라미터 경로 (L719-720)**:
   - 하드코딩된 상대 경로
   - Try-except로 감싸져 있어 직접적 원인은 아님

---

## 다음 세션 권장 사항

### 즉시 시도할 것
1. **Minimal test 작성**:
   ```python
   import sys
   sys.path.insert(0, '/home/jeongho/projects/graduate/YOLO_AirLine')
   from AirLine_assemble_test import _init_airline_models
   print("Initializing...")
   _init_airline_models()
   print("Success!")
   ```

2. **gdb로 crash 지점 파악**:
   ```bash
   gdb python
   (gdb) run test_airline_final.py
   (gdb) bt  # backtrace
   ```

3. **strace로 시스템 콜 추적**:
   ```bash
   strace -o trace.log python test_airline_final.py 2>&1
   tail -100 trace.log
   ```

### 대안 솔루션
1. **AirLine 코드 분리**:
   - `run_airline()` 함수만 별도 파일로 추출
   - 불필요한 import 제거

2. **Docker 컨테이너**:
   - 원저자 환경 재현
   - Ubuntu + Python 3.8 + NumPy 1.x

3. **Pure Python 재구현**:
   - CRG311 없이 AirLine 알고리즘 재구현
   - OpenCV의 HoughLines 또는 LSD 사용

---

## 수정된 파일
- `/home/jeongho/projects/graduate/YOLO_AirLine/AirLine_assemble_test.py`
  - L47: Windows 경로 주석 처리
  - L34-53: Import try-except 추가
  - L128-129: MLP_MODEL_PATH None으로 변경

---

## 확실한 것
- ✅ CRG311.so는 정상 컴파일됨
- ✅ CRG311.desGrow()는 단독으로 작동함
- ✅ DexiNed 모델도 정상 작동
- ✅ Import는 모두 성공

## 불확실한 것
- ❌ 왜 파일 실행 시에만 segfault가 발생하는가?
- ❌ 어느 시점에 정확히 crash하는가? (로그 없음)
- ❌ GPU 메모리 문제인가? (CPU 모드에서도 실패)

---

**작성자**: Claude Code
**마지막 업데이트**: 2025-11-11 21:45
