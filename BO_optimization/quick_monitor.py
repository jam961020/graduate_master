#!/usr/bin/env python
"""
현재 실험 실시간 모니터링 스크립트
"""
import json
import glob
import os
from pathlib import Path

def monitor_current_experiment():
    """현재 실행 중인 실험의 최신 상태를 빠르게 확인"""

    # 가장 최근 run 디렉토리 찾기
    log_dirs = sorted(Path("logs").glob("run_*"), key=os.path.getmtime, reverse=True)

    if not log_dirs:
        print("[ERROR] No experiment logs found")
        return

    latest_run = log_dirs[0]
    print(f"[MONITORING] {latest_run.name}")
    print("=" * 70)

    # Iteration 파일들 찾기
    iter_files = sorted(latest_run.glob("iter_*.json"))

    if not iter_files:
        print("[WAIT] No iteration data yet (initializing...)")
        return

    print(f"[OK] Completed iterations: {len(iter_files)}\n")

    # 최근 10개 또는 전체 출력
    display_count = min(15, len(iter_files))
    print(f"[RESULTS] Latest {display_count} iterations:")
    print("-" * 70)
    print(f"{'Iter':>4} | {'CVaR':>8} | {'Score':>8} | {'Acq Value':>10} | Image")
    print("-" * 70)

    cvars = []
    scores = []
    zero_score_count = 0

    for f in iter_files[-display_count:]:
        try:
            data = json.load(open(f))
            iteration = data['iteration']
            cvar = data['cvar']
            score = data.get('score', 0.0)
            acq_value = data.get('acq_value', 0.0)
            img_idx = data.get('image_idx', '?')

            cvars.append(cvar)
            scores.append(score)
            if score == 0.0:
                zero_score_count += 1

            # Score가 0이면 경고 표시
            warn = " [WARN]" if score == 0.0 else ""

            print(f"{iteration:4d} | {cvar:8.4f} | {score:8.4f} | {acq_value:10.4f} | {img_idx}{warn}")

        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    print("-" * 70)

    # 통계
    if cvars:
        print(f"\n[STATS]")
        print(f"  CVaR - Max: {max(cvars):.4f}, Min: {min(cvars):.4f}, Avg: {sum(cvars)/len(cvars):.4f}")
        print(f"  Score - Avg: {sum(scores)/len(scores):.4f}")
        print(f"  [WARN] Score=0 count: {zero_score_count}/{len(scores)} ({zero_score_count/len(scores)*100:.1f}%)")

        # 추세 분석
        if len(cvars) >= 5:
            recent_5 = cvars[-5:]
            if recent_5[-1] > recent_5[0]:
                trend = "[UP] Improving"
            elif recent_5[-1] < recent_5[0]:
                trend = "[DOWN] Declining"
            else:
                trend = "[FLAT] Plateau"
            print(f"  Recent 5 trend: {trend}")

    # 최고 성능 찾기
    print(f"\n[BEST CVaR]")
    best_cvar = max(cvars)
    best_idx = cvars.index(best_cvar)
    best_file = iter_files[-display_count + best_idx]
    best_data = json.load(open(best_file))
    print(f"  Iteration {best_data['iteration']}: CVaR={best_cvar:.4f}, Score={best_data.get('score', 0):.4f}")

if __name__ == "__main__":
    monitor_current_experiment()
