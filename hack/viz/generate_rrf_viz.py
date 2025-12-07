# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Generate RRF visualization charts using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# 1. RRF Sensitivity Analysis: k parameter vs rank positions
fig, ax = plt.subplots(figsize=(10, 6))
k_values = np.array([10, 20, 40, 60, 80, 100])
ranks = [0, 5, 10]

for rank in ranks:
    scores = 1 / (k_values + rank)
    ax.plot(k_values, scores, marker='o', linewidth=2, markersize=8, 
            label=f'Rank {rank}')

ax.set_xlabel('k Parameter', fontweight='bold')
ax.set_ylabel('RRF Score: 1/(k + rank)', fontweight='bold')
ax.set_title('RRF Score by Rank Position (varying k)\nLower k = stronger top position emphasis', 
             fontweight='bold', pad=15)
ax.legend(title='Rank Position', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_xlim(5, 105)

plt.tight_layout()
plt.savefig(output_dir / 'rrf_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: rrf_sensitivity.png")

# 2. RRF Fusion Example: BM25 + Dense
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Input rankings
documents = ['d1', 'd2', 'd3']
bm25_scores = [12.5, 11.0, 10.5]
dense_scores = [0.7, 0.9, 0.8]

x = np.arange(len(documents))
width = 0.35

bars1 = ax1.bar(x - width/2, bm25_scores, width, label='BM25', 
                color='#00d9ff', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, dense_scores, width, label='Dense', 
                color='#ff6b9d', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Document', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Input Rankings: BM25 vs Dense\n(Incompatible score scales)', 
              fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(documents)
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

# Right: RRF fusion results
rrf_scores = [0.032796, 0.033060, 0.032522]
colors = ['#00ff88' if i == 1 else '#00d9ff' for i in range(3)]

bars = ax2.bar(documents, rrf_scores, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Document', fontweight='bold')
ax2.set_ylabel('RRF Score', fontweight='bold')
ax2.set_title('RRF Fusion Results (k=60)\nd2 wins: appears high in both lists', 
              fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, score) in enumerate(zip(bars, rrf_scores)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.6f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
    if i == 1:
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                'WINNER',
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                color='#00ff88')

plt.tight_layout()
plt.savefig(output_dir / 'rrf_fusion_example.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: rrf_fusion_example.png")

# 3. k parameter comparison table visualization
fig, ax = plt.subplots(figsize=(10, 6))

k_comparison = {
    'k': [10, 60, 100],
    'rank_0': [1/(10+0), 1/(60+0), 1/(100+0)],
    'rank_5': [1/(10+5), 1/(60+5), 1/(100+5)],
    'rank_10': [1/(10+10), 1/(60+10), 1/(100+10)],
}

x = np.arange(len(k_comparison['k']))
width = 0.25

bars1 = ax.bar(x - width, k_comparison['rank_0'], width, label='Rank 0', 
               color='#00ff88', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, k_comparison['rank_5'], width, label='Rank 5', 
               color='#00d9ff', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, k_comparison['rank_10'], width, label='Rank 10', 
               color='#ff6b9d', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('k Parameter', fontweight='bold')
ax.set_ylabel('RRF Score', fontweight='bold')
ax.set_title('RRF Score Comparison: How k Affects Different Rank Positions\nLower k = steeper decay, stronger top position emphasis', 
             fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in k_comparison['k']])
ax.legend(title='Rank Position', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'rrf_k_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Generated: rrf_k_comparison.png")

print("\n✅ All RRF visualizations generated!")

