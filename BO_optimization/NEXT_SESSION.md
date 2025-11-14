# 다음 세션 시작 가이드 (Session 14)

**Date**: 2025-11-14 (Session 13 완료)
**Status**: 🎯 Warm Start 전략 준비 완료
**Priority**: 🚨 HIGH - Warm Start + Top 4 환경 실험

---

## 🔥 즉시 해야 할 일 (1분 체크리스트)

1. [ ] `SESSION_13_CONCLUSION.md` 읽기 (핵심 요약)
2. [ ] `PARADOX_ANALYSIS.md` 읽기 (역설 설명)
3. [ ] Session 13 실험 중단 (필요 시)
4. [ ] `environment_top4.json` 생성
5. [ ] Warm start 구현 시작

---

## 📋 Session 13 요약 (필수 읽기!)

### 결과
- **Best CVaR**: 0.5654 (Iter 9)
- **문제**: 이후 30회 정체, KG correlation = -0.253 (음수!)
- **원인**: ❌ 외삽 아님, ✅ CVaR 예측 실패 (환경 효과 W 예측 부정확)

### 핵심 발견

**역설 발생:**
```
환경 상관 약함 (r=0.12) → CVaR 0.6886 ✅
환경 상관 강함 (r=0.33) → CVaR 0.5654 ❌ (19% 하락!)
```

**외삽 분석:**
```
BO 샘플 vs Initial 거리: 평균 20.7% (MODERATE)
먼 샘플 성능: 0.499 (더 좋음!)
가까운 샘플 성능: 0.456 (나쁨)

→ 외삽이 문제가 아님!
```

**진짜 문제:**
- GP의 환경 효과(W) 예측 부정확
- CVaR 계산에 사용되는 GP 예측 값이 틀림
- KG가 잘못된 목표 최적화

---

## 💡 Opus 제안 전략

### 1. 환경 특징 축소 (6D → 4D)

**Top 4 features (|r| >= 0.35):**
1. local_contrast: r = -0.510
2. clip_rough: r = -0.454
3. brightness: r = -0.364
4. clip_smooth: r = +0.341

**효과:** 14D → 12D, 교차항 48D → 32D (33% 감소)

### 2. Warm Start 전략

**Phase 1 (Warm Start):**
- n_initial = 20
- 환경 없이 **파라미터만 8D 최적화**
- 전체 이미지에서 CVaR 계산
- 목표: 좋은 파라미터 영역 찾기
- 예상 CVaR: 0.62+

**Phase 2 (BO with Environment):**
- iterations = 50
- 파라미터 8D + 환경 4D = 12D
- Phase 1에서 찾은 좋은 X에서 W 관계 학습
- 목표: 환경 고려 fine-tuning
- 예상 최종 CVaR: 0.65+

### 3. Alpha 조절

- alpha = 0.2, 0.3, 0.4 비교 실험

---

## 🚀 즉시 실행 (Step-by-Step)

### Step 1: environment_top4.json 생성 (5분)

```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

python << 'EOF'
import json

# Top 6 로드
with open('environment_top6.json') as f:
    data_top6 = json.load(f)

# Top 4 선택
top4_features = ['local_contrast', 'clip_rough', 'brightness', 'clip_smooth']

data_top4 = {}
for img_name, features in data_top6.items():
    data_top4[img_name] = {
        k: v for k, v in features.items()
        if k in top4_features
    }

# 저장
with open('environment_top4.json', 'w') as f:
    json.dump(data_top4, f, indent=2)

print("✓ Created environment_top4.json")
print(f"  Images: {len(data_top4)}")
print(f"  Features: {list(data_top4[list(data_top4.keys())[0]].keys())}")
EOF
```

---

## 📊 성공 기준

### Warm Start Phase (Phase 1)
- [ ] Best CVaR >= 0.60 (환경 없음)
- [ ] Mean CVaR >= 0.50
- [ ] 20개 샘플 모두 정상 평가

### BO Phase (Phase 2)
- [ ] KG correlation > 0 (양수!)
- [ ] CVaR 꾸준히 증가
- [ ] Final CVaR >= 0.65
- [ ] Session 13 (0.565) 대비 +15% 개선

---

## 📁 참고 파일

### 필수 읽기
1. `SESSION_13_CONCLUSION.md` - 이번 세션 요약
2. `PARADOX_ANALYSIS.md` - 역설 상세 분석
3. `SESSION_13_ANALYSIS.md` - 39 iterations 분석

---

**작성일**: 2025-11-14
**상태**: ✅ Ready to Start
**우선순위**: 🚨 HIGH

**화이팅! Warm start로 돌파하자! 🚀**
