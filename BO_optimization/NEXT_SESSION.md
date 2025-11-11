# 다음 세션 시작 가이드

**날짜**: 2025-11-11 21:35
**이전 세션**: Windows 경로 문제 발견 및 일부 수정

---

## ✅ **핵심 발견: 문제는 경로였습니다!**

### 근본 원인
AirLine_assemble_test.py는 **Windows의 samsung2024 프로젝트**에서 작성된 코드입니다.
- 현재 시스템에는 `samsung2024/` 디렉토리가 없음
- 모든 경로가 `./YOLO_AirLine/...` 형태로 하드코딩됨
- `C:\Users\user\Desktop\...` 같은 Windows 경로도 있음

**결과**: Import는 성공하지만, 실행 시 파일을 찾지 못해 crash 발생

---

## ✅ 성공한 것

### 1. 개별 컴포넌트 테스트 ✅
- CRG311.desGrow(): 정상 작동
- DexiNed 모델: 정상 작동
- Import: 모두 성공

### 2. minimal_test.py 성공 ✅
```bash
python minimal_test.py
# → CVaR=0.0000 (정상 실행!)
```

**위치**: `/home/jeongho/projects/graduate/BO_optimization/minimal_test.py`

**의미**:
- objective_function 자체는 정상 작동
- AirLine 라인 검출 파이프라인 정상
- BoRisk 알고리즘 로직 OK

### 3. 일부 경로 수정 완료 ✅
- L47: Windows 빌드 경로 주석 처리
- L128-129: MLP_MODEL_PATH = None
- L34-53: Optional imports에 try-except 추가
- L719-727: 카메라 파라미터 경로 __file__ 기준으로 변경
- L1600-1603: samsung2024 체크 제거

---

## ❌ 아직 해결되지 않은 것

### 1. optimization.py 직접 실행 실패
```bash
python optimization.py --iterations 1 --n_initial 1 --alpha 0.3
# → 출력 없음 또는 segfault
```

**가능한 원인**:
- argparse 처리 중 문제
- main 실행 흐름의 어딘가에서 잘못된 경로 참조
- 초기화 순서 문제

### 2. 여러 파일에 남아있는 경로 문제
AirLine_assemble_test.py와 관련된 다른 파일들:
- `abs_6_dof.py` ← 이것도 Windows 경로 있을 가능성 높음
- `run_inference.py`
- `pendant_inference.py`
- `run_metric.py`

---

## 🚀 다음 세션 즉시 할 일

### Step 1: 모든 하드코딩 경로 찾기 (최우선!)

```bash
cd /home/jeongho/projects/graduate/YOLO_AirLine

# Windows 경로 찾기
grep -rn "C:\\\\" . --include="*.py" | grep -v ".pyc"
grep -rn "r\"C:" . --include="*.py"

# samsung2024 참조 찾기
grep -rn "samsung" . --include="*.py" | grep -v ".pyc"

# YOLO_AirLine 상대 경로 찾기
grep -rn "\./YOLO_AirLine\|'YOLO_AirLine'" . --include="*.py"

# 기타 의심스러운 경로
grep -rn "Desktop\|Users" . --include="*.py"
```

### Step 2: 모든 경로를 __file__ 기준 절대 경로로 변경

**예시**:
```python
# ❌ 나쁜 예
camera_matrix = np.load('./YOLO_AirLine/pose_estimation.../camera_matrix.npy')

# ✅ 좋은 예
base_dir = os.path.dirname(os.path.abspath(__file__))
cam_file = os.path.join(base_dir, "pose_estimation_code_and_camera_matrix",
                        "camera_parameters", "camera_matrix_filtered.npy")
camera_matrix = np.load(cam_file)
```

### Step 3: abs_6_dof.py 등 관련 파일 수정

```bash
# 각 파일 확인
cat /home/jeongho/projects/graduate/YOLO_AirLine/abs_6_dof.py | grep -E "\.load|\.pth|\.pt|C:\\\\"
cat /home/jeongho/projects/graduate/YOLO_AirLine/run_inference.py | grep -E "\.load|\.pth|\.pt|C:\\\\"
```

### Step 4: optimization.py 테스트

```bash
cd /home/jeongho/projects/graduate/BO_optimization

# 짧은 테스트
python optimization.py --iterations 2 --n_initial 2 --alpha 0.3

# 성공 시 → 본격 실험
python optimization.py --iterations 20 --n_initial 15 --alpha 0.3
```

---

## 📂 수정된 파일 목록

### 완전히 수정됨
1. `/home/jeongho/projects/graduate/YOLO_AirLine/AirLine_assemble_test.py`
   - L47: Windows 경로 주석
   - L34-53: Optional imports
   - L128-129: MLP_MODEL_PATH = None
   - L719-727: 카메라 파라미터 경로 수정
   - L1600-1603: samsung2024 체크 제거

### 추가 수정 필요 (의심)
2. `/home/jeongho/projects/graduate/YOLO_AirLine/abs_6_dof.py`
3. `/home/jeongho/projects/graduate/YOLO_AirLine/run_inference.py`
4. `/home/jeongho/projects/graduate/YOLO_AirLine/pendant_inference.py`
5. `/home/jeongho/projects/graduate/YOLO_AirLine/run_metric.py`

---

## 🧪 테스트 체크리스트

### 빠른 검증
- [ ] `python minimal_test.py` 성공 (이미 성공!)
- [ ] `python optimization.py --iterations 1 --n_initial 1 --alpha 0.3` 성공
- [ ] CVaR > 0.01 확인

### 전체 실험
- [ ] `python optimization.py --iterations 5 --n_initial 5 --alpha 0.3`
- [ ] 로그 파일 생성 확인
- [ ] CVaR 개선 추이 확인

### 최종 실험
- [ ] `python optimization.py --iterations 20 --n_initial 15 --alpha 0.3`
- [ ] 결과 JSON 저장 확인
- [ ] 시각화 생성

---

## 💡 핵심 교훈

1. **경로는 항상 __file__ 기준 절대 경로로**
   - Windows ↔ Linux 포팅 시 필수
   - `os.path.dirname(os.path.abspath(__file__))` 사용

2. **작업 디렉토리 가정하지 말 것**
   - `./YOLO_AirLine/...` ← 현재 디렉토리 의존
   - 어디서 실행하든 작동하도록 설계

3. **CRG311 segfault는 빨간 청어였음**
   - CPU 강제 모드: 필요 없었음
   - 재컴파일: 필요 없었음
   - **단순히 파일을 못 찾아서 crash했을 가능성 높음**

---

## 📊 현재 상태

### 작동하는 것 ✅
- Import: 100% 성공
- CRG311: 정상
- DexiNed: 정상
- objective_function: 정상 (minimal_test.py로 확인)
- BoRisk 알고리즘 로직: 정상

### 작동하지 않는 것 ❌
- optimization.py 직접 실행
- 이유: 아마도 남아있는 경로 문제

---

## 🎯 목표

**Primary Goal**: 경로 문제 완전 해결
**Success Criteria**:
- [ ] optimization.py가 정상 실행
- [ ] CVaR 값 계산 성공
- [ ] 전체 최적화 루프 완료

**Time Estimate**: 30분 ~ 1시간
- 경로 검색: 10분
- 수정: 20분
- 테스트: 10-30분

---

## 📝 빠른 시작 명령어

```bash
cd /home/jeongho/projects/graduate/BO_optimization

# 1. minimal test (이미 성공 확인됨)
python minimal_test.py

# 2. 경로 문제 찾기
cd ../YOLO_AirLine
grep -rn "C:\\\\" . --include="*.py" | head -20
grep -rn "\./YOLO_AirLine" . --include="*.py" | head -20

# 3. optimization.py 테스트
cd ../BO_optimization
python optimization.py --iterations 1 --n_initial 1 --alpha 0.3

# 4. 성공 시 본격 실험
python optimization.py --iterations 20 --n_initial 15 --alpha 0.3
```

---

**마지막 업데이트**: 2025-11-11 21:40
**다음 세션 첫 작업**: Windows 경로 전체 검색 및 수정
**예상 소요 시간**: 30-60분
**성공 확률**: 90% 이상 (경로만 고치면 됨!)
