# Quick Comparison: Environment vs No-Environment

## Session 13 (Current, WITH environment)
- Dimensions: 15D (8D params + 6D env + w sampling)
- n_w: 15 (Sobol sequence)
- Progress: 28/150 iterations
- Best CVaR: 0.5654 (+18.1%)
- KG correlation: -0.253 (NEGATIVE!)
- Issue: GP cannot learn, KG predicts wrong direction

## Previous Success (11/13, WEAK environment correlation)
- Dimensions: Similar setup
- Result: CVaR 0.6886
- Why successful: Environment correlation was WEAK (r=0.12)
  → GP basically ignored environment, optimized parameters only

## Proposed: Pure Parameters (8D, NO environment)
- Dimensions: 8D (params only)
- No n_w needed
- Direct evaluation on ALL images (113)
- Expected: GP learns better, faster convergence

## Recommendation
1. CONTINUE current experiment to 50 iterations (collect data)
2. START new experiment in parallel: 8D params only
3. Compare results at 50 iterations
4. Decide which approach to continue
