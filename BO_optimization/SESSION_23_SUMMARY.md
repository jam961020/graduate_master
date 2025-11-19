# Session 23 Summary (2025-11-19)

## 주요 작업

### 1. 자동 라벨링 완료
- **test2 폴더**: 1031장 이미지 자동 라벨링
- 출력: `ground_truth_auto2.json`
- 사용자가 수동으로 검토/수정 완료

### 2. GT 파일 병합
- `ground_truth.json` (335장) + `ground_truth_auto2.json` (1031장)
- **결과**: `ground_truth_merged.json` (1366장)
- 실제 사용 이미지: `dataset/images/for_BO` (847장) - 나쁜 이미지 삭제됨

### 3. Evaluation 버그 수정
**문제 발견**: 선이 정확히 검출되었는데도 score=0

**원인**:
1. `evaluation.py`가 `ground_truth.json` (335장) 사용 중
   - `ground_truth_merged.json` (1366장) 사용해야 함
2. Threshold가 50px (30px로 해야 함)

**수정 완료**:
```python
# evaluation.py Line 10
GT_FILE = Path(__file__).parent.parent / "dataset" / "ground_truth_merged.json"

# evaluation.py Line 21
def evaluate_lp(detected_coords, image, image_name=None, threshold=30.0, debug=False):
```

### 4. Visualization 업데이트
- `visualization_exploration.py` 수정
- 두 가지 버전 생성:
  1. `visualization_exploration_*.png` - Initial 포함
  2. `visualization_bo_only_*.png` - BO only
- Windows 경로 호환성 수정 (`os.path.basename()`)

---

## 현재 실행 중인 최적화

### 설정
```bash
python optimization.py \
  --image_dir ../dataset/images/for_BO \
  --gt_file ../dataset/ground_truth_merged.json \
  --iterations 200 \
  --n_initial 10 \
  --alpha 0.3 \
  --n_w 15
```

### Initial Sampling 결과 (10회)
| Init | CVaR | Mean |
|------|------|------|
| 1 | 0.3166 | 0.7145 |
| 2 | 0.4210 | 0.7534 |
| 3 | 0.3600 | 0.7177 |
| 4 | **0.5327** | 0.7649 |
| 5 | 0.3846 | 0.7432 |
| 6 | 0.1794 | 0.6819 |
| 7 | 0.3298 | 0.6954 |
| 8 | 0.3744 | 0.7283 |
| 9 | 0.4616 | 0.7384 |

**Initial CVaR 범위**: 0.18 ~ 0.53
**목표**: CVaR 0.7+

---

## LP_r 메트릭 설정

### 현재 설정
- **Threshold**: 30px
- **방식**: 연속(soft) scoring
  - 0px = 1.0점
  - 15px = 0.5점
  - 30px = 0.0점

### 30px 의미
- 1920x1080 기준: 화면 너비의 1.6%
- 부분 매칭에 점수 부여 (너무 엄격하지 않음)

---

## 파일 구조

```
dataset/
├── images/
│   ├── test/           # 기존 이미지 (113장)
│   ├── test2/          # 새 이미지 (1031장)
│   └── for_BO/         # 최종 사용 (847장)
├── ground_truth.json           # 기존 GT (335장)
├── ground_truth_auto2.json     # 자동 라벨링 (1031장)
└── ground_truth_merged.json    # 병합 GT (1366장)
```

---

## 다음 세션 할 일

### 1. 최적화 결과 확인
- 200 iterations 완료 확인
- CVaR 개선 추이 확인
- 최적 파라미터 확인

### 2. Visualization 생성
```bash
python visualization_exploration.py logs/run_20251119_XXXXXX
```

### 3. 논문 Figure 준비
- Initial vs BO 비교 그래프
- 파라미터 수렴 그래프
- Best/Worst 케이스 예시

### 4. (필요시) Threshold 조정
- 현재 30px이 너무 관대하면 15px로 변경 검토
- 단, 너무 엄격하면 최적화 어려워질 수 있음

---

## 실행 명령어

### 결과 확인
```bash
# 최신 run 디렉토리 확인
ls -lt logs/ | head -5

# iter 결과 확인
for f in logs/run_*/iter_*.json; do python -c "import json; d=json.load(open('$f')); print(f'Iter {d[\"iteration\"]:3d}: CVaR={d[\"cvar\"]:.4f}')"; done

# Visualization 생성
python visualization_exploration.py logs/run_20251119_XXXXXX
```

### Labeling Tool
```bash
python labeling_tool.py --image_dir ../dataset/images/for_BO --output ../dataset/ground_truth_merged.json
```

---

## 중요 노트

1. **evaluation.py 수정됨** - GT 파일과 threshold 변경
2. **847장 이미지** 사용 중 (나쁜 이미지 삭제됨)
3. **연속 LP_r** 적용됨 (soft scoring)
4. **밤새 200 iterations** 실행 중

---

**마지막 업데이트**: 2025-11-19 05:30
**다음 세션**: 최적화 결과 확인 및 논문 Figure 생성
