"""
BoRisk 최적화 결과 시각화
교수님 보고용 직관적인 그래프 생성
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 서버 환경용
import numpy as np
from pathlib import Path
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_result(result_file):
    """결과 JSON 파일 로드"""
    with open(result_file, 'r') as f:
        return json.load(f)


def plot_convergence(result_data, output_file):
    """
    수렴 곡선 시각화
    - Best CVaR over iterations
    - 개선 경향성 표시
    """
    history = result_data['history']
    iterations = range(len(history))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. CVaR over iterations
    ax1.plot(iterations, history, 'o-', linewidth=2, markersize=8,
             label='Best CVaR', color='#2E86AB')
    ax1.axhline(y=history[0], color='r', linestyle='--',
                label=f'Initial: {history[0]:.4f}', alpha=0.5)
    ax1.axhline(y=history[-1], color='g', linestyle='--',
                label=f'Final: {history[-1]:.4f}', alpha=0.5)

    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best CVaR Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'BO Convergence (α={result_data["alpha"]})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Improvement over iterations
    improvements = [history[i] - history[i-1] for i in range(1, len(history))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]

    ax2.bar(range(1, len(history)), improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CVaR Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Iteration Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Convergence plot saved: {output_file}")


def plot_parameter_evolution(result_data, output_file):
    """
    파라미터 탐색 궤적 시각화
    - 6D 파라미터 공간 탐색 과정
    """
    param_names = [
        'edgeThresh1', 'simThresh1', 'pixelRatio1',
        'edgeThresh2', 'simThresh2', 'pixelRatio2'
    ]

    # 초기 및 최종 파라미터
    best_params = result_data['best_params']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, param_name in enumerate(param_names):
        ax = axes[idx]

        # 파라미터 범위
        if 'edgeThresh' in param_name:
            param_range = [-23.0, 7.0]
        elif 'simThresh' in param_name:
            param_range = [0.5, 0.99]
        else:  # pixelRatio
            param_range = [0.01, 0.15]

        # 최적 값 표시
        optimal_value = best_params[param_name]

        ax.axvline(x=optimal_value, color='red', linewidth=3,
                   label=f'Optimal: {optimal_value:.3f}')
        ax.axhspan(param_range[0], param_range[1], alpha=0.1, color='blue')

        ax.set_xlim(param_range)
        ax.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Optimal Parameter Configuration',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Parameter plot saved: {output_file}")


def plot_performance_summary(result_data, output_file):
    """
    성능 요약 시각화
    - 초기 vs 최종 성능 비교
    - 개선율 표시
    """
    initial_cvar = result_data['history'][0]
    final_cvar = result_data['history'][-1]
    improvement = result_data['improvement']
    n_images = result_data['n_images']
    alpha = result_data['alpha']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Before/After comparison
    labels = ['Initial\n(Random)', 'Final\n(Optimized)']
    values = [initial_cvar, final_cvar]
    colors = ['#FF6B6B', '#51CF66']

    bars = ax1.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('CVaR Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # 2. Summary statistics
    ax2.axis('off')

    summary_text = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 OPTIMIZATION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Dataset:           {n_images} images
    Risk Level (α):    {alpha:.1%}

    Initial CVaR:      {initial_cvar:.4f}
    Final CVaR:        {final_cvar:.4f}

    Improvement:       {improvement:+.1f}%

    Status:            {'✓ SUCCESS' if improvement > 0 else '✗ NO IMPROVEMENT'}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax2.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Summary plot saved: {output_file}")


def plot_cvar_explanation(alpha, output_file):
    """
    CVaR 개념 설명 시각화 (교수님 보고용)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 샘플 분포 생성
    np.random.seed(42)
    scores = np.random.beta(5, 2, 1000)  # 예시 분포

    # 히스토그램
    n, bins, patches = ax.hist(scores, bins=50, alpha=0.7, color='skyblue',
                                edgecolor='black', density=True)

    # CVaR 영역 표시
    cvar_threshold = np.quantile(scores, alpha)

    # 최악의 α% 영역 강조
    for i, patch in enumerate(patches):
        if bins[i] < cvar_threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)

    ax.axvline(x=cvar_threshold, color='red', linestyle='--', linewidth=2,
               label=f'CVaR threshold (worst {alpha:.0%})')
    ax.axvline(x=scores.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {scores.mean():.3f}')

    ax.set_xlabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'CVaR Concept: Conditional Value at Risk (α={alpha:.0%})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 설명 텍스트
    explanation = (
        f"CVaR focuses on the worst {alpha:.0%} cases\n"
        f"→ Ensures robustness in difficult scenarios\n"
        f"→ Red area: images with lowest performance"
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ CVaR explanation plot saved: {output_file}")


def generate_all_plots(result_file, output_dir=None):
    """
    모든 시각화 생성

    Args:
        result_file: 결과 JSON 파일 경로
        output_dir: 출력 디렉토리 (None이면 results/ 사용)
    """
    result_data = load_result(result_file)

    if output_dir is None:
        output_dir = Path(result_file).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # 파일명 기본값
    base_name = Path(result_file).stem

    print("\n" + "="*60)
    print("Generating Visualization Plots")
    print("="*60)

    # 1. 수렴 곡선
    plot_convergence(
        result_data,
        output_dir / f"{base_name}_convergence.png"
    )

    # 2. 파라미터 진화
    plot_parameter_evolution(
        result_data,
        output_dir / f"{base_name}_parameters.png"
    )

    # 3. 성능 요약
    plot_performance_summary(
        result_data,
        output_dir / f"{base_name}_summary.png"
    )

    # 4. CVaR 설명 (한 번만)
    plot_cvar_explanation(
        result_data['alpha'],
        output_dir / "cvar_explanation.png"
    )

    print("="*60)
    print(f"✓ All plots saved to: {output_dir}")
    print("="*60)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize BoRisk optimization results")
    parser.add_argument("result_file", help="Path to result JSON file")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots")

    args = parser.parse_args()

    generate_all_plots(args.result_file, args.output_dir)
