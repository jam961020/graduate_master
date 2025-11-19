#!/usr/bin/env python
"""Monitor BO optimization progress in real-time"""

import json
import glob
import time
import os
from datetime import datetime

def monitor_progress(log_dir, refresh_interval=10):
    """Monitor progress and print updates"""

    print(f"Monitoring: {log_dir}")
    print(f"Refresh every {refresh_interval} seconds")
    print("-" * 60)

    last_count = 0

    while True:
        try:
            # Find all iteration files
            files = sorted(glob.glob(f'{log_dir}/iter_*.json'))
            current_count = len(files)

            if current_count > last_count:
                # New iteration completed
                latest_file = files[-1]
                with open(latest_file) as f:
                    data = json.load(f)

                # Get file modification time
                mod_time = os.path.getmtime(latest_file)
                mod_datetime = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iter {data['iteration']:3d} COMPLETED")
                print(f"  Time: {mod_datetime}")
                print(f"  CVaR: {data['cvar']:.4f}")
                print(f"  Score: {data['score']:.4f}")
                print(f"  Image: {data['image_idx']}")
                print(f"  Progress: {current_count}/100 ({current_count}%)")

                # Calculate best so far
                all_cvars = []
                for f in files:
                    with open(f) as fp:
                        all_cvars.append(json.load(fp)['cvar'])
                best_cvar = max(all_cvars)
                best_iter = all_cvars.index(best_cvar) + 1

                print(f"  Best CVaR: {best_cvar:.4f} (Iter {best_iter})")

                last_count = current_count

            elif current_count == last_count and current_count > 0:
                # No new iterations, check if stuck
                latest_file = files[-1]
                mod_time = os.path.getmtime(latest_file)
                time_since_update = time.time() - mod_time

                if time_since_update > 300:  # 5 minutes
                    print(f"\n⚠️  WARNING: No new iteration for {time_since_update/60:.1f} minutes")
                    print(f"   Last: Iter {current_count} at {datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')}")

            # Wait before next check
            time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(refresh_interval)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "logs/run_20251117_111151"

    monitor_progress(log_dir)
