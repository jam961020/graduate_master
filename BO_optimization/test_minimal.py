"""최소 테스트 - BoRisk 구조만 테스트"""
import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import AppendFeatures
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

DEVICE = torch.device("cpu")  # CPU만 사용
DTYPE = torch.double

print("Step 1: Creating synthetic data...")
n_initial = 2
n_w = 3
dim_x = 9
dim_w = 6

# 합성 데이터 생성
train_X = torch.randn(n_initial * n_w, dim_x, dtype=DTYPE, device=DEVICE)
train_Y = torch.randn(n_initial * n_w, 1, dtype=DTYPE, device=DEVICE)
w_set = torch.randn(n_w, dim_w, dtype=DTYPE, device=DEVICE)

print(f"  train_X: {train_X.shape}")
print(f"  train_Y: {train_Y.shape}")
print(f"  w_set: {w_set.shape}")

print("\nStep 2: Creating GP with AppendFeatures...")
try:
    gp = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=AppendFeatures(feature_set=w_set)
    )
    print("  GP created successfully!")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nStep 3: Fitting GP...")
try:
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    print("  GP fitted successfully!")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nStep 4: Making predictions...")
try:
    test_X = torch.randn(1, dim_x, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        posterior = gp.posterior(test_X)
        mean = posterior.mean
        print(f"  Prediction mean: {mean.shape}")
        print("  Prediction successful!")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All tests passed!")
