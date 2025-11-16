#!/bin/bash
# Quick test with original LP_r metric (15 iterations, ~30min)

cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

echo "Starting quick test with original LP_r metric..."
echo "Iterations: 15"
echo "n_initial: 5"
echo "alpha: 0.3"
echo "n_w: 15"
echo "max_images: 30"
echo "threshold: 20px (default in evaluate_quality)"
echo ""

python optimization.py \
    --iterations 15 \
    --n_initial 5 \
    --alpha 0.3 \
    --n_w 15 \
    --max_images 30 \
    --env_file environment_top6.json

echo ""
echo "Quick test completed!"
echo "Check results in logs/ directory"
