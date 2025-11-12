#!/bin/bash
echo "Starting optimization.py..."
python optimization.py --iterations 1 --n_initial 1 --alpha 0.3 2>&1 | head -100
echo "Exit code: $?"
