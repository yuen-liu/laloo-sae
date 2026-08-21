"""
Test the case-identity-vs-failure-mode confound for Features 51 and 84: does each feature
fire on ANY poses outside its dominant case (not just report the top-1 case %), and if so,
are those off-home-case firings still bad poses (consistent with detecting a real failure)
or a mix that looks more like noise?

The test set has only 6 distinct protein systems (case-based 70/15/15 split of 36 total),
so this is the sharpest version of the question a reviewer would ask: is Feature 51 truly
silent everywhere except thrombin, or does it fire elsewhere too?

Reuses the exact validated pipeline from case_concentration_check.py (same model, same
pooled train+val selection, same thresholds) -- no retraining.
"""
import sys
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import ConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)

PROJECT_ROOT = '/Users/bridget/Desktop/projects/laloo-sae'
sys.path.insert(0, PROJECT_ROOT)
from src.data_processor import load_processed_data
from src.model import TopKSAE

DATA_DIR = f'{PROJECT_ROOT}/processed_data'
MODEL_DIR = os.path.expanduser('~/Desktop/projects/laloo-sae/models/07_30_26')
RMSD_THRESHOLD = 4.0
K = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latents_normalized, metadata, stats = load_processed_data(DATA_DIR)
metadata = metadata.copy()
metadata['good_pose'] = metadata['rmsd'] < RMSD_THRESHOLD

splits = np.load(f'{DATA_DIR}/splits.npz', allow_pickle=True)
train_idx, val_idx, test_idx = splits['train_idx'], splits['val_idx'], splits['test_idx']

y_train = (~metadata['good_pose'].values[train_idx]).astype(int)
y_val   = (~metadata['good_pose'].values[val_idx]).astype(int)
y_test  = (~metadata['good_pose'].values[test_idx]).astype(int)

X_raw_train = latents_normalized[train_idx]
X_raw_val   = latents_normalized[val_idx]
X_raw_test  = latents_normalized[test_idx]

metadata_test = metadata.iloc[test_idx].reset_index(drop=True)

print('Test set composition:')
print(metadata_test['case_id'].value_counts())
print(f'Total test cases: {metadata_test["case_id"].nunique()} / 36 total systems\n')

summary = torch.load(f'{MODEL_DIR}/training_summary.pkl', map_location='cpu', weights_only=False)
run_stats = summary[K]
best_run_idx = int(np.argmin([r['best_val_loss'] for r in run_stats]))
model = TopKSAE(input_dim=30, hidden_dim=120, k=K, auxk=12, batch_size=256, dead_steps_threshold=2000).to(device)
model.load_state_dict(torch.load(f'{MODEL_DIR}/topksae_k{K}_run{best_run_idx}.pt', map_location=device, weights_only=False))
model.eval()

def extract_activations(model, latents_np, batch_size=2048):
    all_acts = []
    for start in range(0, latents_np.shape[0], batch_size):
        batch = torch.tensor(latents_np[start:start + batch_size], dtype=torch.float32, device=device)
        all_acts.append(model.get_acts(batch).cpu().numpy())
    return np.vstack(all_acts)

acts_train = extract_activations(model, X_raw_train)
acts_val   = extract_activations(model, X_raw_val)
acts_test  = extract_activations(model, X_raw_test)

def best_train_threshold(scores_train, y_bad_train, n_thresholds=200):
    hi = scores_train.max()
    if hi <= 0:
        return None
    hi = min(hi, np.percentile(scores_train[scores_train > 0], 99)) if (scores_train > 0).sum() > 0 else hi
    thresholds = np.linspace(0, hi, n_thresholds + 1)[1:]
    best_f1, best_thr = -1.0, None
    for thr in thresholds:
        flagged = scores_train >= thr
        if flagged.sum() < 10:
            continue
        prec = y_bad_train[flagged].mean()
        rec = y_bad_train[flagged].sum() / y_bad_train.sum()
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr

features_pool = np.concatenate([acts_train, acts_val], axis=0)
y_bad_pool = np.concatenate([y_train, y_val])

FEATURES_OF_INTEREST = [51, 84]
thresholds = {f: best_train_threshold(features_pool[:, f], y_bad_pool) for f in FEATURES_OF_INTEREST}

for feat in FEATURES_OF_INTEREST:
    thr = thresholds[feat]
    firing = acts_test[:, feat] >= thr
    n_firing = int(firing.sum())
    fired = metadata_test.loc[firing].copy()

    print(f'{"="*80}')
    print(f'Feature {feat}  (threshold={thr:.4f}, n_firing={n_firing})')
    print(f'{"="*80}')

    case_counts = fired['case_id'].value_counts()
    print('\nFull case breakdown (ALL cases, not just top-1):')
    for case_id, n in case_counts.items():
        pct = n / n_firing * 100
        print(f'  {case_id:<25} {n:>5,} ({pct:5.1f}%)')

    dominant_case = case_counts.index[0]
    off_home = fired[fired['case_id'] != dominant_case]
    print(f'\nDominant case: {dominant_case} ({case_counts.iloc[0]}/{n_firing} = {case_counts.iloc[0]/n_firing*100:.1f}%)')
    print(f'Off-dominant-case firings: {len(off_home)}/{n_firing} ({len(off_home)/n_firing*100:.1f}%)')

    if len(off_home) > 0:
        off_home_families = set(c.split('_')[0] for c in off_home['case_id'].unique())
        dominant_family = dominant_case.split('_')[0]
        print(f'  Off-home-case protein families: {sorted(off_home_families)}')
        print(f'  Any NON-{dominant_family} family firing? '
              f'{"YES" if off_home_families - {dominant_family} else "NO -- entirely within-family"}')
        print(f'  Off-home-case RMSD: mean={off_home["rmsd"].mean():.2f}, '
              f'median={off_home["rmsd"].median():.2f}, '
              f'min={off_home["rmsd"].min():.2f}, max={off_home["rmsd"].max():.2f}')
        n_bad_off = (~off_home['good_pose']).sum()
        print(f'  Off-home-case bad-pose rate: {n_bad_off}/{len(off_home)} '
              f'({n_bad_off/len(off_home)*100:.1f}%) vs population baseline '
              f'{(~metadata_test["good_pose"]).mean()*100:.1f}%')
    else:
        print('  *** Feature fires on ZERO poses outside its single dominant case. ***')
        dominant_family = dominant_case.split('_')[0]
        same_family_other_case = metadata_test[
            (metadata_test['case_id'].str.startswith(dominant_family)) &
            (metadata_test['case_id'] != dominant_case)
        ]
        if len(same_family_other_case) > 0:
            print(f'  (Other same-family case exists in test set: '
                  f'{same_family_other_case["case_id"].unique().tolist()}, '
                  f'{len(same_family_other_case)} poses -- feature fires on 0 of these too)')
    print()
