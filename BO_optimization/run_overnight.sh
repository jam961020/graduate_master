#!/bin/bash
# Overnight experiment with original LP_r metric (100 iterations, 6-8 hours)

cd /c/Users/user/Desktop/study/task/graduate/graduate_master/BO_optimization

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/overnight_lpr_original_${TIMESTAMP}.log"

echo "Starting overnight experiment with original LP_r metric (STRICT threshold)..."
echo "Iterations: 100"
echo "n_initial: 10"
echo "alpha: 0.3"
echo "n_w: 15"
echo "threshold: 10px (STRICT - improved discrimination)"
echo "Log file: $LOGFILE"
echo ""

nohup python optimization.py \
    --iterations 100 \
    --n_initial 10 \
    --alpha 0.3 \
    --n_w 15 \
    --env_file environment_top6.json \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Monitor with: tail -f $LOGFILE"
echo "Check progress: grep 'Iter.*100' $LOGFILE | tail -10"
echo ""
echo "Expected completion: 6-8 hours"
echo "Good night! 🌙"
