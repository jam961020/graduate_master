# Multi-ROI Strategy Experiment Results

**Date**: 2025-11-14 (Session 11)
**Purpose**: Compare 3 strategies for integrating multiple ROIs

---

## Strategies Compared

1. **first_only** (Baseline): Use only the first longi_WL ROI
2. **average**: Average all ROI features (longi + fillet)
3. **worst_case**: Take max (worst) value for each feature across all ROIs

---

## Results Summary

### Correlation Strength Comparison

| Strategy | Best Feature | Correlation | Strength | CLIP vs Baseline |
|----------|--------------|-------------|----------|------------------|
| **first_only** | clip_smooth | 0.2963 | MODERATE | CLIP +26.4% |
| **average** | local_contrast | -0.3638 | **STRONG** | Baseline +4.0% |
| **worst_case** | local_contrast | **-0.4200** | **STRONG** | Baseline +6.1% |

### Top Features by Strategy

#### Strategy 1: first_only
- clip_smooth: 0.2963 (MODERATE)
- clip_rough: 0.2496 (MODERATE)
- local_contrast: -0.2344 (MODERATE)

#### Strategy 2: average
- local_contrast: -0.3638 (STRONG) **
- clip_rough: 0.3500 (STRONG) **
- brightness: 0.2605 (MODERATE)
- clip_dark: 0.2143 (MODERATE)

#### Strategy 3: worst_case (**WINNER**)
- local_contrast: **-0.4200** (STRONG) **
- clip_rough: **0.3960** (STRONG) **
- brightness: 0.2198 (MODERATE)
- clip_smooth: 0.2108 (MODERATE)

---

## Key Findings

### 1. worst_case is the BEST strategy!

**Correlation improvement:**
- first_only → worst_case: **+41.9% improvement** (0.296 → 0.420)
- MODERATE → **STRONG** correlation

**Why it works:**
- Captures the most difficult ROI characteristics
- Aligns with **BoRisk philosophy** (focus on worst α%)
- Better discriminative power for difficulty

### 2. Important Features

**Top 3 features (worst_case):**
1. `local_contrast`: -0.4200 (lower = better performance)
2. `clip_rough`: 0.3960 (higher = better performance)
3. `brightness`: 0.2198 (higher = better performance)

**Interpretation:**
- **local_contrast**: High local contrast → difficult welding environment
- **clip_rough**: Rough/textured surface → more debris/noise
- **brightness**: Deviation from optimal brightness → harder detection

### 3. Multi-ROI Statistics

- **Total images**: 113
- **Images with multiple ROIs**: 112 (99.1%)
- Most images have 2-3 ROIs (longi_left, longi_right, fillet)

**Feature statistics (worst_case vs first_only):**

| Feature | first_only (mean) | worst_case (mean) | Change |
|---------|------------------|------------------|--------|
| brightness | 0.4538 | 0.5418 | +19.4% |
| contrast | 0.3001 | 0.3764 | +25.4% |
| edge_density | 0.8492 | 0.8852 | +4.2% |
| blur_level | 0.8746 | 0.9251 | +5.8% |

Worst_case captures more extreme difficulty values!

---

## Statistical Analysis

### Correlation Strength Scale
- **STRONG**: |r| > 0.35
- **MODERATE**: 0.25 < |r| <= 0.35
- **WEAK**: 0.15 < |r| <= 0.25
- **NEGLIGIBLE**: |r| <= 0.15

### Strategy Performance
- **worst_case**: 2 STRONG features (local_contrast, clip_rough)
- **average**: 2 STRONG features
- **first_only**: 0 STRONG features

---

## Decision: Use worst_case Strategy

**Reasoning:**
1. **Highest correlation** with performance (|r| = 0.42)
2. **BoRisk-aligned** - focuses on worst-case scenarios
3. **Better discriminative power** - captures extreme difficulty
4. **Robust** - considers all ROIs, not just one

**Next steps:**
1. Use `environment_roi_worst_case.json` for BoRisk experiments
2. Update `optimization.py` to load this environment file
3. Run full BoRisk optimization (30 iterations)
4. Compare with baseline (first_only)

---

## Files Generated

- `environment_roi_first_only.json` - Baseline (113 images)
- `environment_roi_average.json` - Average strategy (113 images)
- `environment_roi_worst_case.json` - **SELECTED** (113 images)

---

## Recommendations for BoRisk

### Environment Vector Configuration
Use `environment_roi_worst_case.json` with 13D features:

**Baseline (9D):**
- brightness, contrast, edge_density
- texture_complexity, blur_level, noise_level
- gradient_strength, sharpness, local_contrast

**CLIP (4D):**
- clip_bright, clip_dark, clip_rough, clip_smooth

**Total: 13D environment vector**

### BoRisk Parameters
```bash
python optimization.py \
    --iterations 30 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_roi_worst_case.json
```

---

**Conclusion**: worst_case strategy provides the strongest correlation with performance and aligns perfectly with BoRisk's risk-aware optimization philosophy. This strategy should significantly improve CVaR optimization by better representing the true difficulty of each image.

**Date**: 2025-11-14
**Status**: ✅ Experiment Complete - Ready for BoRisk Integration
