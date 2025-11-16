"""
환경 특징과 성능(Score)의 연관성 재분석
Session 13 결과 기반
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def main():
    # Session 13 결과 로드
    log_dir = Path('logs/run_20251114_172045')
    iter_files = sorted(log_dir.glob('iter_*.json'))

    print(f'환경-성능 연관성 분석 ({len(iter_files)} iterations)')
    print('='*70)

    # 환경 데이터 로드
    with open('environment_top6.json') as f:
        env_data = json.load(f)

    # 이미지 이름 리스트
    image_names = list(env_data.keys())
    env_features = list(env_data[image_names[0]].keys())

    print(f'\n환경 특징 ({len(env_features)}):')
    for feat in env_features:
        print(f'  - {feat}')

    # Session 13 결과에서 이미지별 score 수집
    image_scores = {}  # {image_idx: [scores]}

    for f in iter_files:
        with open(f) as fp:
            data = json.load(fp)
            img_idx = data.get('image_idx')
            score = data.get('score', 0)

            if img_idx not in image_scores:
                image_scores[img_idx] = []
            image_scores[img_idx].append(score)

    print(f'\n평가된 고유 이미지: {len(image_scores)}개')

    # 데이터셋 로드 (이미지 이름 매핑)
    from optimization import load_dataset
    images_data = load_dataset(
        image_dir="../dataset/images/test",
        gt_file="../dataset/ground_truth.json",
        complete_only=False
    )

    # 이미지별 평균 score 계산 + 환경 특징 매핑
    data_for_analysis = []

    for img_idx, scores in image_scores.items():
        img_name = images_data[img_idx]['name']
        avg_score = np.mean(scores)

        # 환경 특징 가져오기
        if img_name in env_data:
            env_feat = env_data[img_name]

            data_for_analysis.append({
                'image_idx': img_idx,
                'image_name': img_name,
                'avg_score': avg_score,
                'n_evals': len(scores),
                **env_feat
            })

    print(f'분석 가능 이미지: {len(data_for_analysis)}개')

    # 상관관계 분석
    print('\n' + '='*70)
    print('환경 특징 vs 평균 Score 상관관계')
    print('='*70)

    scores = np.array([d['avg_score'] for d in data_for_analysis])

    correlations = {}

    for feat in env_features:
        env_values = np.array([d[feat] for d in data_for_analysis])

        # Pearson correlation
        r_pearson, p_pearson = pearsonr(env_values, scores)

        # Spearman correlation (순위 기반, 비선형 관계 포착)
        r_spearman, p_spearman = spearmanr(env_values, scores)

        correlations[feat] = {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman
        }

        sig_pearson = '***' if p_pearson < 0.001 else '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else ''
        sig_spearman = '***' if p_spearman < 0.001 else '**' if p_spearman < 0.01 else '*' if p_spearman < 0.05 else ''

        print(f'\n{feat}:')
        print(f'  Pearson:  r={r_pearson:+.4f}, p={p_pearson:.4f} {sig_pearson}')
        print(f'  Spearman: r={r_spearman:+.4f}, p={p_spearman:.4f} {sig_spearman}')

    # 상관계수 크기별 정렬
    print('\n' + '='*70)
    print('상관계수 크기순 정렬 (절댓값)')
    print('='*70)

    sorted_feats = sorted(correlations.items(),
                         key=lambda x: abs(x[1]['pearson_r']),
                         reverse=True)

    print(f'\n{"Feature":<20} {"Pearson r":>10} {"Spearman r":>12} {"강도"}')
    print('-'*70)

    for feat, corr in sorted_feats:
        r_p = corr['pearson_r']
        r_s = corr['spearman_r']

        # 상관 강도 판정
        abs_r = abs(r_p)
        if abs_r >= 0.5:
            strength = 'Strong'
        elif abs_r >= 0.3:
            strength = 'Moderate'
        elif abs_r >= 0.1:
            strength = 'Weak'
        else:
            strength = 'Very Weak'

        print(f'{feat:<20} {r_p:+10.4f} {r_s:+12.4f}   {strength}')

    # Visualization
    fig = plt.figure(figsize=(18, 12))

    # Scatter plots (6개 환경 특징)
    for i, feat in enumerate(env_features):
        ax = plt.subplot(3, 2, i+1)

        env_values = np.array([d[feat] for d in data_for_analysis])

        ax.scatter(env_values, scores, alpha=0.6, s=50)

        # 회귀선
        z = np.polyfit(env_values, scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(env_values.min(), env_values.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)

        r = correlations[feat]['pearson_r']
        p_val = correlations[feat]['pearson_p']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel('Avg Score', fontsize=11)
        ax.set_title(f'{feat}\nr={r:+.3f} ({sig})', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Environment Features vs Performance (Session 13)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = 'environment_correlation_analysis.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f'\nVisualization saved: {output_file}')

    # 결론
    print('\n' + '='*70)
    print('결론 및 권장사항')
    print('='*70)

    # 강한 상관 (|r| >= 0.3)
    strong_corr = [feat for feat, corr in correlations.items()
                   if abs(corr['pearson_r']) >= 0.3]

    # 중간 상관 (0.1 <= |r| < 0.3)
    moderate_corr = [feat for feat, corr in correlations.items()
                     if 0.1 <= abs(corr['pearson_r']) < 0.3]

    # 약한 상관 (|r| < 0.1)
    weak_corr = [feat for feat, corr in correlations.items()
                 if abs(corr['pearson_r']) < 0.1]

    print(f'\n강한 상관 (|r| >= 0.3): {len(strong_corr)}개')
    for feat in strong_corr:
        r = correlations[feat]['pearson_r']
        print(f'  - {feat}: r={r:+.4f}')

    print(f'\n중간 상관 (0.1 <= |r| < 0.3): {len(moderate_corr)}개')
    for feat in moderate_corr:
        r = correlations[feat]['pearson_r']
        print(f'  - {feat}: r={r:+.4f}')

    print(f'\n약한 상관 (|r| < 0.1): {len(weak_corr)}개')
    for feat in weak_corr:
        r = correlations[feat]['pearson_r']
        print(f'  - {feat}: r={r:+.4f}')

    # 판정
    print('\n' + '='*70)

    if len(strong_corr) >= 3:
        print('판정: 환경 특징이 성능에 유의미한 영향을 줌')
        print('권장: 환경 특징 유지, Top features 선택 고려')
        print(f'      Top {len(strong_corr)} features 사용 권장')
    elif len(strong_corr) + len(moderate_corr) >= 3:
        print('판정: 환경 특징이 일부 영향을 주지만 약함')
        print('권장: 라벨링 데이터 증가로 상관관계 강화')
        print('      또는 환경 특징 재설계')
    else:
        print('판정: 환경 특징과 성능 간 상관관계 매우 약함')
        print('권장: 환경 특징 제거 고려')
        print('      또는 다른 환경 특징 탐색')

    # 결과 저장
    result = {
        'n_images': len(data_for_analysis),
        'n_evaluations': len(iter_files),
        'correlations': correlations,
        'strong_features': strong_corr,
        'moderate_features': moderate_corr,
        'weak_features': weak_corr
    }

    with open('environment_correlation_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nResult saved: environment_correlation_result.json')

if __name__ == '__main__':
    main()
