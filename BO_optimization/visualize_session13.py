"""
Session 13 (150 iters) Visualization
CVaR vs Score, KG acquisition value 등 분석
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    log_dir = Path('logs/run_20251114_172045')
    iter_files = sorted(log_dir.glob('iter_*.json'))

    print(f'Session 13 Visualization - {len(iter_files)} iterations')
    print('='*70)

    # 데이터 수집
    iterations = []
    cvars = []
    scores = []
    acq_values = []

    for f in iter_files:
        with open(f) as fp:
            data = json.load(fp)
            iterations.append(data.get('iteration', 0))
            cvars.append(data.get('cvar', 0))
            scores.append(data.get('score', 0))
            acq_values.append(data.get('acq_value', 0))

    iterations = np.array(iterations)
    cvars = np.array(cvars)
    scores = np.array(scores)
    acq_values = np.array(acq_values)

    # 통계 출력
    corr_cvar_score = np.corrcoef(cvars, scores)[0, 1]
    corr_acq_cvar = np.corrcoef(acq_values, cvars)[0, 1]

    print(f'\nStatistics:')
    print(f'  CVaR:  Mean={cvars.mean():.4f}, Std={cvars.std():.4f}, Range=[{cvars.min():.4f}, {cvars.max():.4f}]')
    print(f'  Score: Mean={scores.mean():.4f}, Std={scores.std():.4f}, Range=[{scores.min():.4f}, {scores.max():.4f}]')
    print(f'  Acq:   Mean={acq_values.mean():.4f}, Std={acq_values.std():.4f}')
    print(f'\nCorrelations:')
    print(f'  CVaR-Score:  {corr_cvar_score:.4f}')
    print(f'  Acq-CVaR:    {corr_acq_cvar:.4f}')

    # Figure 생성
    fig = plt.figure(figsize=(16, 10))

    # 1. CVaR progression
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(iterations, cvars, 'b-', alpha=0.7, linewidth=1.5, label='CVaR')
    ax1.axhline(cvars.max(), color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(cvars[0], color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('CVaR', fontsize=11)
    ax1.set_title(f'CVaR Progression (Best={cvars.max():.4f} at Iter {iterations[cvars.argmax()]})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 2. Score progression
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(iterations, scores, 'g-', alpha=0.7, linewidth=1.5, label='Score')
    ax2.axhline(scores.max(), color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Score (line_eq)', fontsize=11)
    ax2.set_title(f'Score Progression (Best={scores.max():.4f} at Iter {iterations[scores.argmax()]})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # 3. CVaR vs Score scatter
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(scores, cvars, c=iterations, cmap='viridis', alpha=0.6, s=30)
    ax3.set_xlabel('Score', fontsize=11)
    ax3.set_ylabel('CVaR', fontsize=11)
    ax3.set_title(f'CVaR vs Score (corr={corr_cvar_score:.3f})', fontsize=12)
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Iteration', fontsize=10)

    # 4. Acquisition value
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(iterations[1:], acq_values[1:], 'purple', alpha=0.7, linewidth=1.5, label='KG Value')
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Acquisition Value', fontsize=11)
    ax4.set_title('KG Acquisition Value', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # 5. CVaR histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(cvars, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax5.axvline(cvars.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={cvars.mean():.4f}')
    ax5.set_xlabel('CVaR', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('CVaR Distribution', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Score histogram
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(scores, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax6.axvline(scores.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={scores.mean():.4f}')
    ax6.set_xlabel('Score', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Score Distribution', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Session 13 Analysis ({len(iter_files)} iterations)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 저장
    output_file = 'session13_visualization.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {output_file}')

    # 추가 분석: Best iterations
    print('\n' + '='*70)
    print('Best Iterations Analysis')
    print('='*70)

    best_cvar_idx = cvars.argmax()
    best_score_idx = scores.argmax()
    worst_cvar_idx = cvars.argmin()

    print(f'\nBest CVaR (Iter {iterations[best_cvar_idx]}):')
    print(f'  CVaR:  {cvars[best_cvar_idx]:.4f}')
    print(f'  Score: {scores[best_cvar_idx]:.4f}')
    print(f'  Acq:   {acq_values[best_cvar_idx]:.4f}')

    print(f'\nBest Score (Iter {iterations[best_score_idx]}):')
    print(f'  CVaR:  {cvars[best_score_idx]:.4f}')
    print(f'  Score: {scores[best_score_idx]:.4f}')
    print(f'  Acq:   {acq_values[best_score_idx]:.4f}')

    print(f'\nWorst CVaR (Iter {iterations[worst_cvar_idx]}):')
    print(f'  CVaR:  {cvars[worst_cvar_idx]:.4f}')
    print(f'  Score: {scores[worst_cvar_idx]:.4f}')

    # Convergence 분석
    print('\n' + '='*70)
    print('Convergence Analysis')
    print('='*70)

    # 10회씩 묶어서 평균
    window = 10
    n_windows = len(cvars) // window

    print(f'\nCVaR by window (size={window}):')
    for i in range(min(5, n_windows)):  # 처음 5개 window
        start_idx = i * window
        end_idx = start_idx + window
        window_mean = cvars[start_idx:end_idx].mean()
        print(f'  Iter {start_idx:3d}-{end_idx-1:3d}: {window_mean:.4f}')

    if n_windows > 5:
        print('  ...')
        # 마지막 window
        start_idx = (n_windows - 1) * window
        end_idx = len(cvars)
        window_mean = cvars[start_idx:end_idx].mean()
        print(f'  Iter {start_idx:3d}-{end_idx-1:3d}: {window_mean:.4f}')

if __name__ == '__main__':
    main()
