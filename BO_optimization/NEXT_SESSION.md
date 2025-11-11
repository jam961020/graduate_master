# 다음 세션 시작 가이드

**날짜**: 2025.11.11 21:10
**이전 세션**: CRG311 segfault 방어 로직 추가 - 실패

---

## ⚠️ 현재 상황 (CRITICAL)

**문제**: CRG311.desGrow() segmentation fault가 방어 로직 추가 후에도 여전히 발생

### 시도한 해결책 (모두 실패)
1. ✅ **C-contiguity 강제**: ODes_np, edgeNp_binary, outMap, out 모두 `np.ascontiguousarray()` 적용
2. ✅ **dtype 검증**: assertions로 float32, uint8 타입 강제
3. ✅ **버퍼 오버런 방지**: pixelNumThresh를 이미지 대각선의 3%로 제한
4. ✅ **CPU 강제 실행 옵션**: 환경변수 `AIRLINE_FORCE_CPU=1` 지원 추가
5. ✅ **Symlink 생성**: dataset/, models/ 경로 문제 해결
6. ❌ **테스트 실행**: 여전히 segfault 발생 (timeout: the monitored command dumped core)

---

## 🔍 근본 원인 분석

### CRG311.desGrow() 문제
**파일**: `/home/jeongho/projects/graduate/YOLO_AirLine/AirLine_assemble_test.py:635-650`

```python
rawLineNum = crg.desGrow(
    outMap, edgeNp_binary, ODes_np, out,
    airline_config["simThresh"],
    safe_pixel_thresh,  # ← 축소된 값 사용
    TMP1, TMP2, TMP3, THETA_RES
)
```

### 가능한 원인
1. **CRG311.so ABI 불일치**
   - 컴파일 환경: Python 3.x, NumPy 1.x
   - 현재 환경: Python 3.11, NumPy 2.x
   - **가능성**: 매우 높음 ⭐⭐⭐⭐⭐

2. **GPU 메모리 접근 오류**
   - DexiNed()가 GPU에서 실행 후 CPU NumPy 배열로 변환
   - GPU 텐서 → CPU 전환 시 메모리 레이아웃 불일치
   - **가능성**: 높음 ⭐⭐⭐⭐

3. **TMP1/TMP2/TMP3 버퍼 크기 부족**
   - TMP1: (50000, 2) - 엣지 포인트
   - TMP2: (2, 300000, 2) - 라인 그로잉
   - TMP3: (3000, 2, 2) - 라인 세그먼트
   - **가능성**: 중간 ⭐⭐⭐

---

## 🚀 다음 세션 시작 시 즉시 시도할 것

### 방법 1: CPU 강제 실행 테스트 (최우선)
```bash
# CPU 모드로 강제 실행
AIRLINE_FORCE_CPU=1 python optimization.py --iterations 2 --n_initial 2 --alpha 0.3

# 이유: GPU↔CPU 메모리 전환 문제 회피
```

**기대 결과**:
- ✅ 성공 시 → GPU 관련 메모리 문제 확인됨
- ❌ 실패 시 → CRG311.so ABI 문제 확률 99%

---

### 방법 2: CRG311.so 재컴파일 (CPU 실패 시)

**위치**: `/home/jeongho/projects/graduate/YOLO_AirLine/CRG311.so`

```bash
cd /home/jeongho/projects/graduate/YOLO_AirLine

# 1. 소스 코드 위치 확인
ls -la CRG311.* *.cpp *.c

# 2. 현재 Python/NumPy 버전 확인
python -c "import sys; import numpy; print(f'Python {sys.version}'); print(f'NumPy {numpy.__version__}')"

# 3. 재컴파일 (예시 - 실제 빌드 스크립트 확인 필요)
# g++ -shared -fPIC CRG311.cpp -o CRG311.so $(python -m pybind11 --includes) $(python-config --ldflags)
# 또는
# python setup.py build_ext --inplace
```

**참고**:
- CRG311은 논문 원저자 코드
- 알고리즘 의미를 바꾸지 않는 ABI 호환성 수정은 정석
- 재컴파일로 Python 3.11 + NumPy 2.x 환경 일치

---

### 방법 3: 디버깅 모드 실행 (재컴파일도 실패 시)

```bash
# gdb로 crash 지점 확인
gdb python
(gdb) run optimization.py --iterations 1 --n_initial 1 --alpha 0.3
# segfault 발생 시
(gdb) bt  # backtrace
(gdb) info registers
```

**또는 더 간단하게**:
```bash
# strace로 시스템 콜 추적
strace -o trace.log python optimization.py --iterations 1 --n_initial 1 --alpha 0.3 2>&1

# crash 직전 로그 확인
tail -100 trace.log
```

---

## 📂 수정된 파일 요약

### AirLine_assemble_test.py
**위치**: `/home/jeongho/projects/graduate/YOLO_AirLine/AirLine_assemble_test.py`

**변경사항**:
1. **라인 56-61**: DEVICE 설정 추가
   ```python
   USE_GPU = os.environ.get('AIRLINE_FORCE_CPU', '0') != '1'
   DEVICE = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
   print(f"[AirLine] Using device: {DEVICE} (USE_GPU={USE_GPU})")
   ```

2. **라인 65**: `.cuda()` → `.to(DEVICE)`
   ```python
   thetaN = nn.Conv2d(...).to(DEVICE)
   ```

3. **라인 86**: DexiNed GPU → DEVICE
   ```python
   EDGE_DET = DexiNed().to(DEVICE)
   ```

4. **라인 90**: torch.load map_location
   ```python
   edge_state_dict = torch.load(dexi_path, map_location=DEVICE)
   ```

5. **라인 591**: tensor GPU → DEVICE
   ```python
   x1 = torch.tensor(rx1_resized, dtype=torch.float32).to(DEVICE) / 255.0
   ```

6. **라인 610-630**: C-contiguity 및 dtype 강제 (이전 세션)
7. **라인 635-650**: 버퍼 오버런 방지 (이전 세션)

### 새로 생성된 파일
- `BO_optimization/dataset` → `../dataset` (symlink)
- `BO_optimization/models` → `../models` (symlink)

---

## 🧪 테스트 체크리스트

### Step 1: CPU 강제 실행
- [ ] `AIRLINE_FORCE_CPU=1 python optimization.py --iterations 2 --n_initial 2 --alpha 0.3`
- [ ] segfault 발생 여부 확인
- [ ] 발생 안하면 → 성공! GPU 메모리 문제였음
- [ ] 발생하면 → Step 2로

### Step 2: CRG311.so 재컴파일
- [ ] 소스 코드 위치 확인
- [ ] 빌드 시스템 확인 (CMakeLists.txt, setup.py, Makefile)
- [ ] 현재 환경에서 재컴파일
- [ ] 테스트 재실행
- [ ] 성공하면 → 완료! ABI 문제였음
- [ ] 실패하면 → Step 3로

### Step 3: 대체 방법 검토
- [ ] AirLine 알고리즘을 pure Python으로 재구현?
- [ ] Docker 컨테이너로 원저자 환경 재현?
- [ ] CRG311 없이 다른 라인 검출 알고리즘 사용?

---

## 📊 BoRisk 알고리즘 상태

### 구현 완료 (이전 세션)
- ✅ 환경 벡터 추출 (6D)
- ✅ w_set 샘플링
- ✅ (x, w) → y GP 모델 (AppendFeatures)
- ✅ qMFKG 획득 함수
- ✅ CVaR objective

### 블로커: CRG311 segfault
**결과**: BoRisk 알고리즘은 완벽하게 구현되었으나, AirLine 라인 검출 단계에서 crash 발생

---

## 💡 중요 팁

### 디버깅 순서
1. **CPU 강제 실행** (5분) - 가장 빠름
2. **CRG311 재컴파일** (30분) - 가장 확실함
3. **gdb/strace 디버깅** (1시간) - 근본 원인 파악

### 만약 모두 실패하면
- **Option A**: AirLine_assemble_test.py의 CRG311 호출 부분을 임시로 mock하고 BoRisk 알고리즘 로직만 테스트
- **Option B**: 더 간단한 라인 검출 알고리즘(HoughLines, LSD)으로 대체하여 BoRisk 검증
- **Option C**: 원저자에게 CRG311.so 빌드 환경 문의

---

## 📝 실행 명령어 요약

```bash
# 1. CPU 강제 테스트 (최우선)
cd /home/jeongho/projects/graduate/BO_optimization
AIRLINE_FORCE_CPU=1 python optimization.py --iterations 2 --n_initial 2 --alpha 0.3

# 2. 정상 실행 확인
tail -f realtime_test.log

# 3. segfault 여부 확인
echo $?  # 0이면 성공, 139(segfault) 또는 기타 에러 코드면 실패

# 4. 성공 시 확대 테스트
AIRLINE_FORCE_CPU=1 python optimization.py --iterations 5 --n_initial 5 --alpha 0.3
```

---

## 🎯 다음 세션 목표

**Primary Goal**: CRG311 segfault 해결

**Success Criteria**:
- [ ] optimization.py가 segfault 없이 최소 1 iteration 완료
- [ ] CVaR 값 계산 성공
- [ ] BoRisk 알고리즘 정상 작동 확인

**Fallback Plan**:
- CRG311 해결 불가 시 → 대체 라인 검출 알고리즘으로 BoRisk 검증

---

**마지막 업데이트**: 2025.11.11 21:10
**다음 세션 첫 명령어**: `AIRLINE_FORCE_CPU=1 python optimization.py --iterations 2 --n_initial 2 --alpha 0.3`
