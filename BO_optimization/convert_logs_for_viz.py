"""
로그 디렉토리의 iter_*.json 파일들을 읽어서
visualize_exploration.py가 기대하는 형식으로 변환
"""
import json
from pathlib import Path
import numpy as np
import argparse

def convert_logs_to_viz_format(log_dir):
    """
    logs/run_XXXXX/ 디렉토리의 iter_*.json 파일들을 읽어서
    visualize_exploration.py 형식으로 변환
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"Error: {log_dir} does not exist")
        return None

    # iter_*.json 파일들 찾기
    iter_files = sorted(log_path.glob("iter_*.json"))

    if not iter_files:
        print(f"Error: No iter_*.json files found in {log_dir}")
        return None

    print(f"Found {len(iter_files)} iteration files")

    # 데이터 수집
    history = []
    best_cvar = -np.inf
    best_params = None
    best_iter = 0

    for iter_file in iter_files:
        with open(iter_file, 'r') as f:
            data = json.load(f)

        cvar = data.get('cvar', 0.0)
        history.append(cvar)

        # Best 추적
        if cvar > best_cvar:
            best_cvar = cvar
            best_params = data.get('parameters', {})
            best_iter = data.get('iteration', len(history))

    # 결과 딕셔너리 생성
    result = {
        'history': history,
        'best_params': best_params,
        'best_cvar': best_cvar,
        'best_iteration': best_iter,
        'total_iterations': len(history)
    }

    # 저장
    output_file = log_path / "visualization_data.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OK] Converted {len(history)} iterations")
    print(f"[OK] Best CVaR: {best_cvar:.4f} at iteration {best_iter}")
    print(f"[OK] Saved to: {output_file}")

    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert log directory to visualization format")
    parser.add_argument("log_dir", help="Path to log directory (e.g., logs/run_20251114_044828)")
    args = parser.parse_args()

    convert_logs_to_viz_format(args.log_dir)
