"""
Check whether the case-concentration pattern seen in Features 51/84 (the paper's two
spotlighted interpretability examples) is general across the K=8 4A combo filter's full
candidate pool, or specific to those two features.

Reproduces threshold_4_valselect.ipynb's Section 12 pipeline (pooled train+val candidate
selection + combo search) standalone, validates against the paper's published K=8 headline
numbers, then reports per-feature dominant-case concentration for the whole candidate pool.
No retraining -- uses the existing best-of-5-seeds K=8 checkpoint.
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

# ---------- 1. Load data, splits, model (mirrors notebook cells 1-8) ----------
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

summary = torch.load(f'{MODEL_DIR}/training_summary.pkl', map_location='cpu', weights_only=False)
run_stats = summary[K]
best_run_idx = int(np.argmin([r['best_val_loss'] for r in run_stats]))
model = TopKSAE(input_dim=30, hidden_dim=120, k=K, auxk=12, batch_size=256, dead_steps_threshold=2000).to(device)
model.load_state_dict(torch.load(f'{MODEL_DIR}/topksae_k{K}_run{best_run_idx}.pt', map_location=device, weights_only=False))
model.eval()
print(f'Loaded K={K} run {best_run_idx} (val_loss={run_stats[best_run_idx]["best_val_loss"]:.4f})')

def extract_activations(model, latents_np, batch_size=2048):
    all_acts = []
    for start in range(0, latents_np.shape[0], batch_size):
        batch = torch.tensor(latents_np[start:start + batch_size], dtype=torch.float32, device=device)
        all_acts.append(model.get_acts(batch).cpu().numpy())
    return np.vstack(all_acts)

acts_train = extract_activations(model, X_raw_train)
acts_val   = extract_activations(model, X_raw_val)
acts_test  = extract_activations(model, X_raw_test)
print(f'Extracted activations: train={acts_train.shape}, val={acts_val.shape}, test={acts_test.shape}')

# ---------- 2. Pooled train+val selection pipeline (mirrors notebook cells 24/26/28) ----------
def threshold_stats(scores_train, scores_test, y_bad_train, y_bad_test, n_thresholds=200):
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
    if best_thr is None:
        return None
    flagged_test = scores_test >= best_thr
    if flagged_test.sum() == 0:
        return None
    prec = y_bad_test[flagged_test].mean()
    rec = y_bad_test[flagged_test].sum() / y_bad_test.sum()
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'n_flagged': int(flagged_test.sum()), 'pct_flagged': flagged_test.mean() * 100}

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

def eval_union(acts, y_bad, feats, thresholds):
    flags = {f: (acts[:, f] >= thresholds[f]).astype(int) for f in feats}
    records = []
    for r in range(1, len(feats) + 1):
        for combo in combinations(feats, r):
            combined = np.zeros(len(y_bad), dtype=int)
            for feat in combo:
                combined |= flags[feat]
            flagged = combined.astype(bool)
            if flagged.sum() == 0:
                continue
            prec = y_bad[flagged].mean()
            rec = y_bad[flagged].sum() / y_bad.sum()
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            good_kept = (~flagged)[y_bad == 0].mean()
            records.append({'subset': "+".join(str(f) for f in combo), 'n_flagged': int(flagged.sum()),
                             'pct_flagged': flagged.mean() * 100, 'precision': prec, 'recall': rec,
                             'f1': f1, 'good_kept': good_kept})
    return pd.DataFrame(records)

def eval_combo(acts, y_bad, combo, thresholds):
    flagged = np.zeros(len(y_bad), dtype=bool)
    for feat in combo:
        flagged |= (acts[:, feat] >= thresholds[feat])
    n = int(flagged.sum())
    prec = y_bad[flagged].mean() if n > 0 else float('nan')
    rec = y_bad[flagged].sum() / y_bad.sum() if n > 0 else float('nan')
    f1 = 2 * prec * rec / (prec + rec + 1e-9) if n > 0 else float('nan')
    good_kept = (~flagged)[y_bad == 0].mean()
    return {'subset': "+".join(str(f) for f in combo), 'n_flagged': n, 'pct_flagged': flagged.mean() * 100,
            'precision': prec, 'recall': rec, 'f1': f1, 'good_kept': good_kept}

features_pool = np.concatenate([acts_train, acts_val], axis=0)
y_bad_pool = np.concatenate([y_train, y_val])
print(f'\nPooled train+val selection set: {len(y_bad_pool):,} poses ({(1 - y_bad_pool.mean())*100:.1f}% good)')

recs = []
for feat in range(120):
    r = threshold_stats(features_pool[:, feat], features_pool[:, feat], y_bad_pool, y_bad_pool)
    if r:
        recs.append({'feature': feat, **r})
rebuilt_df = pd.DataFrame(recs)
print(f'K={K}: {len(rebuilt_df)} features with a valid threshold')

GOOD_KEPT_MIN = 0.80
N_CANDIDATES = 16
candidates = (rebuilt_df[rebuilt_df['recall'] > 0.02]
              .nlargest(N_CANDIDATES, 'precision')['feature'].astype(int).tolist())
print(f'\nTop {len(candidates)} candidates by precision (pooled train+val, recall>0.02): {candidates}')

thresholds = {f: best_train_threshold(features_pool[:, f], y_bad_pool) for f in candidates}
val_abl_df = eval_union(features_pool, y_bad_pool, candidates, thresholds)
feasible = val_abl_df[val_abl_df['good_kept'] >= GOOD_KEPT_MIN]
best = feasible.loc[feasible['recall'].idxmax()] if len(feasible) > 0 else val_abl_df.loc[val_abl_df['good_kept'].idxmax()]
winning_combo = [int(f) for f in best['subset'].split('+')]
print(f'\nWinning combo (pooled train+val selection): {winning_combo}')

test_row = eval_combo(acts_test, y_test, winning_combo, thresholds)
print(f'\n--- VALIDATION: applied once to TEST ---')
print(f"  {test_row['pct_flagged']:.1f}% flagged, precision={test_row['precision']:.3f}, "
      f"recall={test_row['recall']:.3f}, good_kept={test_row['good_kept']*100:.1f}%")
print(f"  Paper's published K=8 4A headline: 16.5% flagged, precision=0.803, recall=0.210, good_kept=91.1%")

# ---------- 3. Case-concentration breakdown for the full candidate pool ----------
print(f'\n{"="*80}')
print('CASE-CONCENTRATION BREAKDOWN (test set, all candidate features)')
print(f'{"="*80}')

case_records = []
for feat in candidates:
    thr = thresholds[feat]
    firing = acts_test[:, feat] >= thr
    n_firing = int(firing.sum())
    if n_firing == 0:
        continue
    precision = y_test[firing].mean()
    top_cases = metadata_test.loc[firing, 'case_id'].value_counts()
    top_case = top_cases.index[0]
    top_case_n = int(top_cases.iloc[0])
    top_case_pct = top_case_n / n_firing * 100
    case_records.append({
        'feature': feat, 'in_winning_combo': feat in winning_combo,
        'n_firing': n_firing, 'precision': precision,
        'dominant_case': top_case, 'dominant_case_pct': top_case_pct,
        'n_distinct_cases': metadata_test.loc[firing, 'case_id'].nunique(),
    })

case_df = pd.DataFrame(case_records).sort_values('dominant_case_pct', ascending=False).reset_index(drop=True)
print(case_df.to_string(index=False, float_format='{:.1f}'.format))

n_combo = case_df['in_winning_combo'].sum()
combo_df = case_df[case_df['in_winning_combo']]
print(f'\n{"-"*80}')
print(f'Winning combo has {n_combo} feature(s): {winning_combo}')
print(f'Of the {len(case_df)} candidate-pool features (test set, n_firing>0):')
for thresh in [90, 75, 50]:
    n_above = (case_df['dominant_case_pct'] >= thresh).sum()
    print(f'  {n_above}/{len(case_df)} concentrate >={thresh}% of firings in a single dominant case')
n_distinct_dominant = case_df['dominant_case'].nunique()
print(f'  {n_distinct_dominant} distinct dominant case_ids represented across the {len(case_df)} candidates')
print(f'\nWinning-combo-only breakdown:')
print(combo_df.to_string(index=False, float_format='{:.1f}'.format))

out_path = f'{PROJECT_ROOT}/output/4A/case_concentration_k8_candidates.csv'
case_df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
