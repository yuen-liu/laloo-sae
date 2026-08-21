"""
Regenerate feature_51_activation_dist.png / feature_84_activation_dist.png
as separate, larger-font, higher-DPI panels for the paper.

Reproduces the exact computation in threshold_4_valselect.ipynb cell 42
(K=8 model, activation split by the 2Å native-pose criterion), but:
  - saves each feature as its own file (paper.tex includes them separately)
  - uses much larger fonts/DPI so labels stay legible after LaTeX
    downscales to ~0.26-0.31\\linewidth in the 6-panel Figure 1 layout
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = '/Users/bridget/Desktop/projects/laloo-sae'
FEATURES_NPZ = f'{PROJECT_ROOT}/output/4A/features_for_paper.npz'
METADATA_CSV = f'{PROJECT_ROOT}/output/4A/metadata.csv'
SPLITS_NPZ   = f'{PROJECT_ROOT}/processed_data/splits.npz'
FIG_DIR      = f'{PROJECT_ROOT}/paper/figures'

feat_data = np.load(FEATURES_NPZ, allow_pickle=True)
test_idx  = feat_data['test_idx']

splits = np.load(SPLITS_NPZ, allow_pickle=True)
assert np.array_equal(splits['test_idx'], test_idx), "test_idx mismatch between splits.npz and features_for_paper.npz"

metadata      = pd.read_csv(METADATA_CSV)
metadata_test = metadata.iloc[test_idx].reset_index(drop=True)
good_mask_2A  = (metadata_test['rmsd'].values <= 2.0)

features_k8 = feat_data['k8']  # [N_test, 120]

SPOTLIGHT_FEATURES = [
    (51, 'Feature 51 (catastrophic, thrombin)', f'{FIG_DIR}/feature_51_activation_dist.png'),
    (84, 'Feature 84 (geometric, uPA)',          f'{FIG_DIR}/feature_84_activation_dist.png'),
]

plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 19,
})

for feat, title, out_path in SPOTLIGHT_FEATURES:
    a = features_k8[:, feat]
    active = a > 0
    a_act = a[active]
    y_act = good_mask_2A[active]

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    bins = np.linspace(0, a_act.max() if len(a_act) else 1.0, 60)
    ax.hist(a_act[~y_act], bins=bins, density=False, alpha=0.75, color='#0072B2',
            label=f'bad (n={(~y_act).sum():,})', edgecolor='none')
    if y_act.sum() > 0:
        ax.hist(a_act[y_act], bins=bins, density=False, alpha=0.75, color='#E6B800',
                label=f'good (n={y_act.sum():,})', edgecolor='none')
    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(top=ymax * 6)  # headroom so the legend doesn't overlap the peak
    ax.set_xlabel('Activation value')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=22, pad=10)
    ax.legend(fontsize=19, loc='upper right', framealpha=0.95)
    ax.tick_params(width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}  (n_active={active.sum():,})')
