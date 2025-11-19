# Session 22 Summary (2025-11-18)

## Main Issue Resolved: Score=0 Bug

### Problem
- Optimization running but `score=0` in results despite visible line detections
- Example: `iter_011.json` showed `score: 0.0` with `cvar: 0.3514`

### Root Cause
- `evaluation.py` was using `ground_truth.json` with only **113 images**
- User had created `ground_truth_auto.json` with **335 images** (manual labeling)
- 222 images had no ground truth -> returned `score=0`

### Fix
- User had already swapped files: `ground_truth_auto.json` -> `ground_truth.json`
- Verification confirmed: `ground_truth.json` now contains 335 images

---

## Test Results

### Quick Test (10 images, 5 iterations)
- Initial CVaR: 0.6412
- Final CVaR: 0.8231
- **Improvement: +28.4%**
- All scores calculated properly!

### Full Test (335 images, 30 iterations, n_w=15)
**Run: `run_20251118_190314`**

| Metric | Value |
|--------|-------|
| Initial Sampling Best | 0.5713 |
| BO Iter 1 | 0.6790 |
| BO Iter 30 (Final) | 0.7250 |
| **Best CVaR** | **0.7251** (Iter 24) |
| Total Improvement | **+26.9%** |

### Best Parameters Found
```
AirLine Q:
  edgeThresh1: -21.69
  simThresh1: 0.838
  pixelRatio1: 0.135

AirLine QG:
  edgeThresh2: -11.53
  simThresh2: 0.977
  pixelRatio2: 0.070

RANSAC:
  weight_q: 13.30
  weight_qg: 18.16
```

---

## Visualization Update

### Modified `visualization_exploration.py`
- Added initial sampling CVaR display in Panel 1
- Initial sampling shown with negative x-axis values (-9 to 0)
- BO iterations shown with positive x-axis (1 to 30)
- Vertical dotted line separates phases
- Fixed Windows path compatibility (`os.path.basename()`)

### Graph Now Shows
- **Gray line**: Initial random sampling (CVaR: 0.33 ~ 0.57)
- **Blue line**: BO iterations (CVaR: 0.68 ~ 0.73)
- Clear visualization of BO's improvement over random search

---

## Key Insights

1. **BO vs Random**: First BO iteration (0.6790) already better than best random sample (0.5713)
2. **Consistent Improvement**: Every few iterations shows CVaR improvement
3. **Convergence**: Graph shows system still improving at iter 30, not fully converged yet
4. **Score Distribution**: Most evaluations now return valid scores (not 0)

---

## Files Modified
- `visualization_exploration.py`: Added initial sampling display, fixed Windows paths

## Files Created
- `results/visualization_exploration_run_20251118_190314.png`: Updated visualization with initial sampling

---

## Next Steps for Paper

### Immediate (Before Friday Deadline)
1. **Expand to 1000 images**
   - Add more images to dataset
   - Update `ground_truth.json` with labels
   - Run full optimization (50-100 iterations)

2. **Generate Paper Figures**
   - Use updated visualization with initial sampling
   - Shows dramatic improvement from random to BO

3. **Comparison Experiments**
   - Baseline: Random search
   - Competitor: Standard BO (EI)
   - Ours: BoRisk with CVaR

### Command for Final Run
```bash
cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization
conda activate weld2024_mk2

# Full experiment with more images
python optimization.py --iterations 100 --n_initial 20 --alpha 0.3 --n_w 20
```

---

## Session Statistics
- **Duration**: ~2 hours
- **Main Achievement**: Fixed score=0 bug, achieved 26.9% CVaR improvement
- **Visualization**: Now shows full optimization journey including initial sampling

---

**Next Session**: Expand dataset to 1000 images and run final experiment for paper
