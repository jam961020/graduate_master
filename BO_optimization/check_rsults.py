"""
실험 결과 확인 및 중단된 실험 복구
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def list_experiments():
    """
    모든 실험 결과 리스트
    """
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found!")
        return
    
    # latest 파일들 찾기
    latest_files = sorted(results_dir.glob("*_latest.json"))
    
    if not latest_files:
        print("No experiments found!")
        return
    
    print("\n" + "="*80)
    print("Available Experiments")
    print("="*80)
    print(f"{'#':<4} {'Timestamp':<20} {'Status':<12} {'Iter':<8} {'Best CVaR':<12} {'Improvement':<12}")
    print("-"*80)
    
    experiments = []
    for idx, filepath in enumerate(latest_files, 1):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        timestamp = filepath.stem.replace('_latest', '').split('_')[-2:]
        timestamp_str = f"{timestamp[0]}_{timestamp[1]}"
        status = data['status']
        current_iter = data.get('current_iteration', 'N/A')
        best_cvar = data.get('best_cvar', 0)
        improvement = data.get('improvement', 0)
        
        print(f"{idx:<4} {timestamp_str:<20} {status:<12} {current_iter:<8} {best_cvar:<12.4f} {improvement:<+12.2f}%")
        experiments.append(filepath)
    
    print("="*80)
    return experiments


def show_experiment_details(exp_file):
    """
    실험 상세 정보 표시
    """
    with open(exp_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("Experiment Details")
    print("="*60)
    print(f"\nStatus: {data['status']}")
    print(f"Metric: {data['metric']}")
    print(f"Images: {data['n_images']}")
    print(f"Env samples: {data['n_env_samples']}")
    print(f"K neighbors: {data['k_neighbors']}")
    
    if data['status'] == 'running':
        print(f"\nCurrent iteration: {data['current_iteration']}")
    else:
        print(f"\nTotal iterations: {data.get('iterations', 'N/A')}")
    
    print(f"\nCurrent/Final CVaR: {data.get('current_cvar', data.get('final_cvar', 0)):.4f}")
    print(f"Best CVaR: {data['best_cvar']:.4f}")
    print(f"Initial CVaR: {data['history'][0]:.4f}")
    print(f"Improvement: {data['improvement']:+.2f}%")
    
    print(f"\nBest Parameters:")
    for key, val in data['best_params'].items():
        print(f"  {key:15s}: {val:8.4f}")
    
    # 히스토리 그래프
    if len(data['history']) > 1:
        plot_progress(data)


def plot_progress(data):
    """
    진행 상황 그래프
    """
    history = np.array(data['history'])
    
    plt.figure(figsize=(12, 5))
    
    # 1. CVaR 진화
    ax1 = plt.subplot(1, 2, 1)
    iterations = np.arange(len(history))
    ax1.plot(iterations, history, 'b-', marker='o', markersize=4, linewidth=2)
    ax1.axhline(y=history[0], color='r', linestyle='--', alpha=0.5, label='Initial')
    ax1.axhline(y=history.max(), color='g', linestyle='--', alpha=0.5, label='Best')
    
    best_iter = np.argmax(history)
    ax1.plot(best_iter, history[best_iter], 'r*', markersize=15, label=f'Best (iter {best_iter})')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('CVaR', fontsize=12)
    ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 개선도
    ax2 = plt.subplot(1, 2, 2)
    improvements = history - history[0]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax2.bar(iterations, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Improvement from Initial', fontsize=12)
    ax2.set_title('Cumulative Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def get_all_iterations(base_filename):
    """
    특정 실험의 모든 iteration 파일 찾기
    """
    results_dir = Path("results")
    base = base_filename.replace('_latest.json', '')
    
    # init 파일들
    init_files = sorted(results_dir.glob(f"{base}_init*.json"))
    # iter 파일들
    iter_files = sorted(results_dir.glob(f"{base}_iter*.json"))
    
    return init_files, iter_files


def recover_last_checkpoint(exp_file):
    """
    마지막 체크포인트로부터 복구
    """
    with open(exp_file, 'r') as f:
        data = json.load(f)
    
    if data['status'] == 'completed':
        print("Experiment is already completed!")
        return
    
    print("\n" + "="*60)
    print("Recovery Information")
    print("="*60)
    print(f"\nLast saved iteration: {data['current_iteration']}")
    print(f"Best CVaR so far: {data['best_cvar']:.4f}")
    print(f"History length: {len(data['history'])}")
    
    # 모든 iteration 파일 확인
    base_filename = exp_file.name
    init_files, iter_files = get_all_iterations(base_filename)
    
    print(f"\nSaved checkpoints:")
    print(f"  Initial samples: {len(init_files)}")
    print(f"  BO iterations: {len(iter_files)}")
    print(f"  Total evaluations: {len(init_files) + len(iter_files)}")
    
    print(f"\nYou can resume from iteration {data['current_iteration']}")
    print(f"All progress has been saved!")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("Experiment Manager")
    print("="*60)
    
    experiments = list_experiments()
    
    if not experiments:
        sys.exit(0)
    
    print("\nOptions:")
    print("  1. Show details of an experiment")
    print("  2. Check recovery status")
    print("  3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        exp_num = int(input("Enter experiment number: "))
        if 1 <= exp_num <= len(experiments):
            show_experiment_details(experiments[exp_num - 1])
        else:
            print("Invalid experiment number!")
    
    elif choice == '2':
        exp_num = int(input("Enter experiment number: "))
        if 1 <= exp_num <= len(experiments):
            recover_last_checkpoint(experiments[exp_num - 1])
        else:
            print("Invalid experiment number!")
    
    elif choice == '3':
        print("Exiting...")
    
    else:
        print("Invalid option!")