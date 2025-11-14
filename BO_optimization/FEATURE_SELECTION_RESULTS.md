# Feature Selection Results - Top 6 Environment Features

**Date**: 2025-11-14 (Session 11)
**Goal**: Select best 6 features from 15D candidates

---

## Experiments Performed

### 1. Multi-ROI Strategy Comparison
- **Strategies**: first_only, average, worst_case
- **Features**: 15D (9 baseline + 4 CLIP + 2 quality metrics)
- **Winner**: **worst_case** (r = -0.42)

### 2. PSNR/SSIM Quality Metrics
- Added advanced image quality metrics:
  - `psnr_score`: Peak Signal-to-Noise Ratio
  - `ssim_score`: Structural Similarity Index
- **Reference**: Gaussian blurred version

---

## Results Summary

### PSNR/SSIM Performance

| Strategy | psnr_score (r) | ssim_score (r) | Strength |
|----------|---------------|----------------|----------|
| first_only | -0.1517 | -0.0867 | NEGLIGIBLE |
| average | -0.0969 | 0.0236 | NEGLIGIBLE |
| worst_case | -0.0800 | 0.0783 | NEGLIGIBLE |

**Conclusion**: PSNR/SSIM did NOT improve correlation. Using Gaussian blur as reference doesn't capture welding image difficulty well.

---

## Top 6 Features (worst_case strategy)

### Ranking by Absolute Correlation

| Rank | Feature | Correlation | Strength | Type | Interpretation |
|------|---------|-------------|----------|------|----------------|
| 1 | **local_contrast** | **-0.4200** | **STRONG** | Baseline | Lower local contrast = Better performance |
| 2 | **clip_rough** | **0.3960** | **STRONG** | CLIP | Higher roughness detected = Better performance (?) |
| 3 | **brightness** | 0.2198 | MODERATE | Baseline | Less deviation from optimal = Better |
| 4 | **clip_smooth** | 0.2108 | MODERATE | CLIP | Smoother surface = Better |
| 5 | **gradient_strength** | -0.2052 | MODERATE | Baseline | Lower gradient = Better |
| 6 | **edge_density** | 0.1976 | WEAK | Baseline | Optimal edge density = Better |

**Alternative 6th feature:**
- `contrast` (0.1956) - Very close to edge_density

---

## Selected 6D Environment Vector

```python
SELECTED_FEATURES = [
    'local_contrast',      # r = -0.42 (STRONG)
    'clip_rough',          # r =  0.40 (STRONG)
    'brightness',          # r =  0.22 (MODERATE)
    'clip_smooth',         # r =  0.21 (MODERATE)
    'gradient_strength',   # r = -0.21 (MODERATE)
    'edge_density'         # r =  0.20 (WEAK)
]
```

**Composition:**
- **Baseline features**: 4 (local_contrast, brightness, gradient_strength, edge_density)
- **CLIP features**: 2 (clip_rough, clip_smooth)

**Total dimensions**: 6D

---

## Comparison: 13D vs 6D

### 13D (All features - worst_case)
- Best correlation: -0.4200
- 2 STRONG features
- 3 MODERATE features
- Contains noise from weak features

### 6D (Selected - worst_case)
- Best correlation: -0.4200 (same top feature)
- 2 STRONG features
- 3 MODERATE features
- 1 WEAK feature
- **Benefits**:
  - 54% dimension reduction (13D → 6D)
  - Faster GP training
  - Less overfitting risk
  - Easier interpretation

---

## Feature Interpretation

### 1. local_contrast (-0.42)
- **Definition**: Standard deviation within local windows
- **Meaning**: High local contrast = Complex texture/shadows
- **Why negative**: More contrast → More difficult → Lower performance

### 2. clip_rough (0.40)
- **Definition**: CLIP similarity to "rough textured surface with debris"
- **Meaning**: Semantic roughness detection
- **Why positive**: Higher CLIP score correlates with better performance (needs investigation!)

### 3. brightness (0.22)
- **Definition**: Deviation from optimal brightness (128)
- **Meaning**: Too bright or too dark
- **Why positive**: Less deviation → Better

### 4. clip_smooth (0.21)
- **Definition**: CLIP similarity to "smooth clean surface"
- **Meaning**: Semantic cleanliness
- **Why positive**: Cleaner surface → Better detection

### 5. gradient_strength (-0.21)
- **Definition**: Mean Sobel gradient magnitude
- **Meaning**: Edge strength
- **Why negative**: Stronger gradients → Noisier → Harder

### 6. edge_density (0.20)
- **Definition**: Deviation from optimal edge density (0.1-0.3)
- **Meaning**: Too many or too few edges
- **Why positive**: Closer to optimal → Better

---

## Statistical Significance

**Sample size**: 44 images evaluated
**Correlation threshold**: |r| > 0.3 for practical significance

**Top 2 features** (r > 0.35):
- local_contrast: -0.42
- clip_rough: 0.40

Both exceed threshold → **Statistically significant** at α=0.05

---

## Recommendations

### For BoRisk Optimization

1. **Use worst_case strategy** with 6D features
2. **Create new environment file**:
   ```bash
   python extract_environment_top6.py \
       --strategy worst_case \
       --output environment_top6.json
   ```

3. **Update optimization.py**:
   ```python
   # Load 6D environment
   env_file = "environment_top6.json"
   w_dim = 6  # Updated from 13
   ```

4. **Benefits**:
   - Faster BO iterations (less GP overhead)
   - Better generalization (less overfitting)
   - Easier to interpret results

---

## Files Generated

- `environment_roi_first_only_v2.json` - 15D with PSNR/SSIM
- `environment_roi_average_v2.json` - 15D with PSNR/SSIM
- `environment_roi_worst_case_v2.json` - 15D with PSNR/SSIM

**Next step**: Create `environment_top6.json` with selected 6 features

---

## Future Work

1. **Investigate clip_rough positive correlation**
   - Why does roughness correlate with better performance?
   - Is this counter-intuitive result real or artifact?

2. **Test alternative PSNR/SSIM references**
   - Use different reference (e.g., adaptive histogram equalized version)
   - Self-similarity metrics instead of external reference

3. **Evaluate on more images**
   - Current: 44/113 images evaluated
   - Goal: 100+ images for robust statistics

4. **Feature engineering**
   - Combination features (e.g., brightness × contrast)
   - Spatial features (e.g., ROI position)

---

**Date**: 2025-11-14
**Status**: ✅ Feature Selection Complete - Ready for 6D BoRisk Optimization
**Next**: Create environment_top6.json and run full BoRisk experiment
