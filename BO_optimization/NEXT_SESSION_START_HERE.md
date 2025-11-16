# 다음 세션 시작 가이드

**날짜**: 2025-11-16 작성
**이전 세션**: Session 14 (LP_r 구현 완료)

---

## 🎯 빠른 요약

### 지난 세션 성과
- ✅ **LP_r 원본 구현** (AirLine 논문)
- ✅ **Correlation 개선**: -0.19 → **0.41** (Moderate!)
- ✅ **Quick test 성공** (15 iters)

### LP_r이 뭔가요?
```
LP_r = GT 픽셀 중 검출된 선으로부터 threshold 이내에 있는 비율
     = Recall (GT coverage)

예: LP_r = 0.88 → GT의 88%가 20px 이내에 검출됨
```

**중요**:
- "Line Precision"이지만 실제로는 **Recall**
- RANSAC이 단일 선 선택 → Over-detection 없음
- → LP_r만으로 충분!

---

## 📋 다음 세션 TODO (우선순위)

### 1️⃣ 라벨링 증가 (가장 중요!)

**현재**: 113개 이미지
**목표**: 200개 이미지

**방법 A: 자동 라벨링**
```bash
# AirLine으로 자동 추출
python auto_labeling.py --input_dir ../dataset/images/test
```

**방법 B: 수동 라벨링**
```bash
# Labeling tool 사용
python labeling_tool.py
```

**예상 시간**: 1-2시간 (이미지당 1분)

---

### 2️⃣ Overnight 실험 (100 iterations)

**Quick test가 r=0.41로 promising!**

**실행**:
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
bash run_overnight.sh
```

**또는 직접**:
```bash
nohup python optimization.py \
    --iterations 100 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_top6.json \
    > logs/overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**기대**: r > 0.5, CVaR > 0.92

---

### 3️⃣ (선택) Threshold 실험

**목적**: 최적 tolerance 찾기

- threshold=10 (엄격)
- threshold=20 (현재)
- threshold=50 (관대)

---

## 📊 Quick Test 결과 (참고)

```
Iterations: 15
CVaR: 0.82 → 0.89 (+8.7%)
Best CVaR: 0.91 (Iter 11)

CVaR-Score correlation: 0.41 (Moderate!)
  vs Session 13: -0.19 (실패)
  vs Overnight: 0.07 (실패)

Perfect score: 46.7% (여전히 높지만 개선 중)
```

**의미**: 올바른 방향! 더 긴 실험 필요.

---

## 📁 주요 파일

### 읽어야 할 문서
1. **SESSION_14_COMPLETE.md** - 전체 세션 보고서 (이 내용의 상세 버전)
2. **LP_METRIC_ANALYSIS.md** - LP metric 분석

### 실행 스크립트
3. **run_overnight.sh** - Overnight 실험

### 코드
4. **evaluation.py** - LP_r 구현 (수정됨)

---

## 🚀 바로 시작하기

### Step 1: 지난 세션 확인 (5분)
```bash
# SESSION_14_COMPLETE.md 읽기
cat SESSION_14_COMPLETE.md
```

### Step 2: 라벨링 작업 (1-2시간)
```bash
# 자동 또는 수동
```

### Step 3: Overnight 실험 시작 (1분)
```bash
bash run_overnight.sh
```

### Step 4: 자기 (6-8시간)
```bash
# 내일 아침 결과 확인
```

---

## 💡 핵심만 기억하기

1. **LP_r = Recall** (GT coverage)
2. **Correlation 0.41** (개선됨!)
3. **라벨링 증가 필수** (200개 목표)
4. **Overnight 실험** (100 iters)

---

**다음 세션 화이팅! 🎓**
