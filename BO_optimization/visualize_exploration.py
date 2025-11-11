"""
BO 파라미터 탐색 과정 시각화
실험 후 results/에서 실행
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_bo_exploration(result_file):
    """
    BO 탐색 과정 시각화
    """
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # 데이터 추출
    history = np.array(data['history'])
    if len(history) == 0:
        print('Empty history in result file:', result_file)
        return
    if len(history) == 1:
        history = np.array([history[0], history[0]])
        best_params = data['best_params']
    
    # 전체 탐색 파라미터 로드 (파일에 저장되어 있어야 함)
    # 없으면 best_params만 사용
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. CVaR 진화 (더 상세하게)
    ax1 = plt.subplot(3, 3, 1)
    iterations = np.arange(len(history))
    ax1.plot(iterations, history, 'b-', linewidth=2, marker='o', markersize=6, alpha=0.7)
    
    # 초기/최종 표시
    ax1.axhline(y=history[0], color='r', linestyle='--', alpha=0.5, label='Initial')
    ax1.axhline(y=history[-1], color='g', linestyle='--', alpha=0.5, label='Final')
    
    # Best 표시
    best_iter = np.argmax(history)
    ax1.plot(best_iter, history[best_iter], 'r*', markersize=20, 
             label=f'Best (iter {best_iter})')
    
    # 개선 구간 표시
    for i in range(1, len(history)):
        if history[i] > history[i-1]:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('CVaR (F1 Score)', fontsize=12)
    ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 개선도 (누적)
    ax2 = plt.subplot(3, 3, 2)
    cumulative_best = np.maximum.accumulate(history)
    improvements = cumulative_best - history[0]
    ax2.fill_between(iterations, 0, improvements, alpha=0.3, color='green')
    ax2.plot(iterations, improvements, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cumulative Improvement', fontsize=12)
    ax2.set_title('Total Improvement Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 반복별 개선도 (막대)
    ax3 = plt.subplot(3, 3, 3)
    iter_improvements = np.diff(cumulative_best, prepend=history[0])
    colors = ['green' if x > 0 else 'gray' for x in iter_improvements]
    ax3.bar(iterations, iter_improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Step Improvement', fontsize=12)
    ax3.set_title('Per-Iteration Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-6. 파라미터 값 추이 (있으면)
    param_names = ['edgeThresh1', 'simThresh1', 'pixelRatio1', 
                   'edgeThresh2', 'simThresh2', 'pixelRatio2']
    
    # Best 파라미터만 표시
    ax4 = plt.subplot(3, 3, 4)
    ax4.axis('off')
    param_text = "Best Parameters\n" + "="*40 + "\n\n"
    for key, val in best_params.items():
        param_text += f"{key:15s}: {val:8.4f}\n"
    ax4.text(0.1, 0.5, param_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    # 5. 통계
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis('off')
    stats_text = f"""
Optimization Statistics
{'='*40}

Initial CVaR:     {history[0]:.4f}
Final CVaR:       {history[-1]:.4f}
Best CVaR:        {history.max():.4f}
Improvement:      {(history[-1] - history[0]):+.4f}
Improvement %:    {(history[-1] - history[0])/history[0]*100:+.2f}%

Best at iter:     {best_iter}
# Improvements:   {np.sum(iter_improvements > 0)}
# Stagnations:    {np.sum(iter_improvements == 0)}
    """
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    # 6. CVaR 분포 (히스토그램)
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(history, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax6.axvline(x=history[0], color='r', linestyle='--', linewidth=2, label='Initial')
    ax6.axvline(x=history[-1], color='g', linestyle='--', linewidth=2, label='Final')
    ax6.set_xlabel('CVaR Value', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('CVaR Distribution', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. 이동 평균 (추세)
    ax7 = plt.subplot(3, 3, 7)
    window = min(5, len(history) // 3)
    if window > 1:
        moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
        ax7.plot(iterations, history, 'b-', alpha=0.3, label='Raw')
        ax7.plot(iterations[window-1:], moving_avg, 'r-', linewidth=2, 
                label=f'{window}-iter Moving Avg')
        ax7.set_xlabel('Iteration', fontsize=12)
        ax7.set_ylabel('CVaR', fontsize=12)
        ax7.set_title('Smoothed Trend', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Exploitation vs Exploration
    ax8 = plt.subplot(3, 3, 8)
    # 현재 best 대비 새 샘플 점수 비교
    exploitation = []
    exploration = []
    current_best = history[0]
    for i, score in enumerate(history):
        if score >= current_best * 0.95:  # 95% 이상이면 exploitation
            exploitation.append(i)
        else:
            exploration.append(i)
        current_best = max(current_best, score)
    
    ax8.scatter(exploitation, [history[i] for i in exploitation], 
               c='green', s=50, alpha=0.6, label='Exploitation')
    ax8.scatter(exploration, [history[i] for i in exploration], 
               c='blue', s=50, alpha=0.6, label='Exploration')
    ax8.set_xlabel('Iteration', fontsize=12)
    ax8.set_ylabel('CVaR', fontsize=12)
    ax8.set_title('Exploration vs Exploitation', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 개선 속도 (derivative)
    ax9 = plt.subplot(3, 3, 9)
    if len(history) > 2:
        derivatives = np.gradient(cumulative_best)
        ax9.plot(iterations, derivatives, 'purple', linewidth=2)
        ax9.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax9.fill_between(iterations, 0, derivatives, where=(derivatives>0), 
                        alpha=0.3, color='green')
        ax9.set_xlabel('Iteration', fontsize=12)
        ax9.set_ylabel('Improvement Rate', fontsize=12)
        ax9.set_title('Learning Speed', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = Path(result_file).parent / f"{Path(result_file).stem}_exploration.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_exploration.py results/borisk_*.json")
        
        # 자동으로 최신 파일 찾기
        results_dir = Path("results")
        if results_dir.exists():
            json_files = sorted(results_dir.glob("borisk_*.json"), 
                              key=lambda x: x.stat().st_mtime, 
                              reverse=True)
            if json_files:
                print(f"\nUsing latest result: {json_files[0]}")
                visualize_bo_exploration(json_files[0])
            else:
                print("No result files found in results/")
        sys.exit(1)
    
    result_file = sys.argv[1]
    visualize_bo_exploration(result_file)