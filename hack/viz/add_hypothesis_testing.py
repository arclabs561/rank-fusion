# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
#     "scipy>=1.10.0",
# ]
# ///
"""
Add hypothesis testing to existing RRF visualizations.
Enhances statistical rigor with t-tests, ANOVA, effect sizes.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# Load real evaluation data
eval_json = output_dir.parent.parent / "evals" / "eval_results.json"
if not eval_json.exists():
    eval_json = output_dir.parent.parent.parent / "rank-fusion" / "evals" / "eval_results.json"

if eval_json.exists():
    with open(eval_json) as f:
        eval_results = json.load(f)
    
    # Extract method metrics
    methods_data = {}
    for scenario in eval_results:
        methods = scenario.get('methods', {})
        for method_name, method_data in methods.items():
            if method_name not in methods_data:
                methods_data[method_name] = {'ndcg_at_10': [], 'precision_at_10': [], 'mrr': []}
            metrics = method_data.get('metrics', {})
            if 'ndcg_at_10' in metrics:
                methods_data[method_name]['ndcg_at_10'].append(metrics['ndcg_at_10'])
            if 'precision_at_10' in metrics:
                methods_data[method_name]['precision_at_10'].append(metrics['precision_at_10'])
            if 'mrr' in metrics:
                methods_data[method_name]['mrr'].append(metrics['mrr'])
    
    # Focus on key methods
    key_methods = ['rrf', 'combsum', 'combmnz', 'borda']
    available_methods = {k: v for k, v in methods_data.items() if k in key_methods}
    
    if available_methods:
        # Hypothesis Testing Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_test = ['ndcg_at_10', 'precision_at_10', 'mrr']
        metric_names = ['NDCG@10', 'Precision@10', 'MRR']
        
        for idx, (metric, name) in enumerate(zip(metrics_to_test, metric_names)):
            ax = axes[idx]
            
            # Get data for all methods
            method_data_list = []
            method_labels = []
            for method in key_methods:
                if method in available_methods:
                    data = available_methods[method].get(metric, [])
                    if data:
                        method_data_list.append(data)
                        method_labels.append(method.upper())
            
            if len(method_data_list) >= 2:
                # Box plot
                bp = ax.boxplot(method_data_list, tick_labels=method_labels,
                               patch_artist=True, showmeans=True)
                
                colors = ['#00ff88', '#00d9ff', '#ff6b9d', '#ffd93d']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Perform ANOVA
                if len(method_data_list) >= 3:
                    f_stat, p_value = stats.f_oneway(*method_data_list)
                    
                    # Add statistical test results
                    test_text = f'ANOVA:\nF-statistic: {f_stat:.3f}\np-value: {p_value:.2e}\n'
                    if p_value < 0.05:
                        test_text += 'Significant difference ✓'
                    else:
                        test_text += 'No significant difference'
                    
                    ax.text(0.7, 0.95, test_text, transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=10, verticalalignment='top', fontweight='bold',
                           family='monospace')
                
                # Pairwise t-tests (RRF vs others)
                if 'rrf' in available_methods and len(method_data_list) >= 2:
                    rrf_data = available_methods['rrf'].get(metric, [])
                    if rrf_data:
                        pairwise_results = []
                        for method in key_methods:
                            if method != 'rrf' and method in available_methods:
                                other_data = available_methods[method].get(metric, [])
                                if other_data and len(rrf_data) == len(other_data):
                                    t_stat, p_val = stats.ttest_rel(rrf_data, other_data)
                                    pairwise_results.append(f'{method.upper()}: p={p_val:.3f}')
                        
                        if pairwise_results:
                            pairwise_text = 'Pairwise vs RRF:\n' + '\n'.join(pairwise_results[:3])
                            ax.text(0.05, 0.95, pairwise_text, transform=ax.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                                   fontsize=9, verticalalignment='top', fontweight='bold',
                                   family='monospace')
            
            ax.set_ylabel(name, fontweight='bold')
            ax.set_title(f'{name}: Method Comparison\nWith hypothesis testing', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rrf_hypothesis_testing.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Generated: rrf_hypothesis_testing.png")
        
        # Effect Size Analysis
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Compute Cohen's d for RRF vs each method
        if 'rrf' in available_methods:
            rrf_ndcg = available_methods['rrf'].get('ndcg_at_10', [])
            if rrf_ndcg:
                effect_sizes = []
                method_names = []
                
                for method in key_methods:
                    if method != 'rrf' and method in available_methods:
                        other_ndcg = available_methods[method].get('ndcg_at_10', [])
                        if other_ndcg and len(rrf_ndcg) == len(other_ndcg):
                            # Cohen's d
                            pooled_std = np.sqrt((np.var(rrf_ndcg) + np.var(other_ndcg)) / 2)
                            if pooled_std > 0:
                                d = (np.mean(rrf_ndcg) - np.mean(other_ndcg)) / pooled_std
                                effect_sizes.append(d)
                                method_names.append(method.upper())
                
                if effect_sizes:
                    colors_effect = ['#00ff88' if d > 0 else '#ff4757' for d in effect_sizes]
                    bars = ax.barh(method_names, effect_sizes, color=colors_effect, alpha=0.8,
                                  edgecolor='black', linewidth=1.5)
                    
                    ax.axvline(0, color='black', linestyle='-', linewidth=2)
                    ax.axvline(0.2, color='gray', linestyle='--', linewidth=1, label='Small effect')
                    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, label='Medium effect')
                    ax.axvline(0.8, color='gray', linestyle='--', linewidth=1, label='Large effect')
                    
                    # Add value labels
                    for bar, d in zip(bars, effect_sizes):
                        ax.text(d, bar.get_y() + bar.get_height()/2,
                               f'{d:.3f}', ha='left' if d > 0 else 'right',
                               va='center', fontweight='bold', fontsize=10)
                    
                    ax.set_xlabel("Cohen's d (Effect Size)", fontweight='bold', fontsize=12)
                    ax.set_ylabel('Method', fontweight='bold', fontsize=12)
                    ax.set_title("Effect Size: RRF vs Other Methods\nCohen's d: |d|<0.2 small, 0.2-0.8 medium, >0.8 large", 
                               fontweight='bold', pad=15)
                    ax.legend(frameon=True, fancybox=True, shadow=True)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / 'rrf_effect_size.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print("✅ Generated: rrf_effect_size.png")
        
        print("\n✅ Hypothesis testing visualizations generated!")

