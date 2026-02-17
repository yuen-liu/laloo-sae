"""Data processing pipeline for protein-ligand SAE training"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .data_loader import load_pickle_data
from .config import LATENT_DIM, EXCLUSION_LIST


def find_case_folders(root_dir, exclusion_list=None):
    """
    Find all case folders with optimization and initialization data.
    
    Returns:
        List of dicts with case_id, opt_dir, init_dir
    """
    exclusion_list = exclusion_list or []
    root_dir = Path(root_dir)
    
    all_folders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
    opt_folders = [f for f in all_folders if f.name.endswith('_optimization')]
    
    cases = []
    for opt_folder in opt_folders:
        case_id = opt_folder.name.replace('_optimization', '')
        
        # Check exclusion
        if any(ex in case_id for ex in exclusion_list):
            print(f"[SKIP] Excluded: {case_id}")
            continue
        
        init_folder = root_dir / f"{case_id}_initialization"
        
        if not init_folder.exists():
            print(f"[!] Missing initialization folder for {case_id}")
            continue
        
        cases.append({
            'case_id': case_id,
            'opt_dir': str(opt_folder),
            'init_dir': str(init_folder)
        })
    
    print(f"Found {len(cases)} valid cases")
    return cases


def process_all_cases(root_dir, output_dir, exclusion_list=None, latent_dim=LATENT_DIM):
    """
    Main processing pipeline: load all data and save to NPZ.
    
    Returns:
        stats, latents_array, metadata_df
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cases = find_case_folders(root_dir, exclusion_list)
    
    all_latents = []
    all_metadata = []
    case_info = {}
    
    print("\nProcessing cases...")
    for case in tqdm(cases):
        case_id = case['case_id']
        opt_dir = case['opt_dir']
        init_dir = case['init_dir']
        
        # Load data
        latents, rmsd, energy, gen = load_pickle_data(opt_dir, latent_dim)
        
        if len(latents) == 0:
            print(f"[!] No valid data for {case_id}")
            continue
        
        if latents.shape[1] != latent_dim:
            print(f"[!] Wrong latent dimension for {case_id}: {latents.shape[1]}")
            continue
        
        # Create metadata
        n_samples = len(latents)
        metadata = pd.DataFrame({
            'case_id': [case_id] * n_samples,
            'sample_idx': np.arange(n_samples),
            'rmsd': rmsd,
            'energy': energy,
            'generation': gen,
        })
        
        all_latents.append(latents)
        all_metadata.append(metadata)
        
        # Store case info
        mae_files = glob.glob(os.path.join(init_dir, "*.mae*"))
        groups_files = glob.glob(os.path.join(init_dir, "*groups*.pkl"))
        good_poses = (rmsd < 2.0).sum()
        
        case_info[case_id] = {
            "template_path": mae_files[0] if mae_files else None,
            "groups_path": groups_files[0] if groups_files else None,
            "opt_dir": opt_dir,
            "init_dir": init_dir,
            "num_poses": n_samples,
            "good_poses": int(good_poses),
            "rmsd_min": float(rmsd.min()),
            "rmsd_max": float(rmsd.max()),
        }
    
    if not all_latents:
        raise RuntimeError("No data loaded — check paths or pickle schemas.")
    
    # Consolidate
    print("\nConsolidating data...")
    latents_array = np.vstack(all_latents)
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    metadata_df['global_idx'] = np.arange(len(metadata_df))
    
    # Compute stats
    print("Computing statistics...")
    stats = compute_stats(latents_array, metadata_df)
    
    # Normalize
    latents_normalized = (latents_array - stats['mean']) / (stats['std'] + 1e-8)
    
    # Save
    print("Saving data...")
    save_to_npz(output_dir, latents_array, latents_normalized, metadata_df, stats)
    
    # Save case info
    with open(output_dir / 'case_info.pkl', 'wb') as f:
        pickle.dump(case_info, f)
    
    print("\n✓ Processing complete!")
    return stats, latents_array, metadata_df


def compute_stats(latents, metadata):
    """Compute and print dataset statistics"""
    stats = {
        'mean': latents.mean(axis=0),
        'std': latents.std(axis=0),
        'n_samples': len(latents),
        'n_cases': metadata['case_id'].nunique(),
        'rmsd_percentiles': {
            'p25': float(np.percentile(metadata['rmsd'], 25)),
            'p50': float(np.percentile(metadata['rmsd'], 50)),
            'p75': float(np.percentile(metadata['rmsd'], 75)),
            'p90': float(np.percentile(metadata['rmsd'], 90)),
        },
        'energy_percentiles': {
            'p10': float(np.percentile(metadata['energy'], 10)),
            'p25': float(np.percentile(metadata['energy'], 25)),
            'p50': float(np.percentile(metadata['energy'], 50)),
            'p75': float(np.percentile(metadata['energy'], 75)),
        }
    }
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {stats['n_samples']:,}")
    print(f"Total cases: {stats['n_cases']}")
    print(f"Samples per case (avg): {stats['n_samples']/stats['n_cases']:.1f}")
    print(f"\nRMSD percentiles: {stats['rmsd_percentiles']}")
    print(f"Energy percentiles: {stats['energy_percentiles']}")
    print(f"\nLatent statistics:")
    print(f"  Mean range: [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]")
    print(f"  Std range: [{stats['std'].min():.3f}, {stats['std'].max():.3f}]")
    
    # Good vs poor poses
    good_poses = (metadata['rmsd'] < 2.0).sum()
    print(f"\nPose Quality:")
    print(f"  Good poses (< 2Å): {good_poses} ({good_poses/len(metadata)*100:.1f}%)")
    print(f"  Poor poses (≥ 2Å): {len(metadata)-good_poses} ({(1-good_poses/len(metadata))*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return stats


def save_to_npz(output_dir, latents_raw, latents_norm, metadata, stats):
    """Save dataset to compressed NPZ format"""
    filepath = output_dir / 'dataset.npz'
    
    # Save all arrays in one compressed file
    np.savez_compressed(
        filepath,
        latents_raw=latents_raw,
        latents_normalized=latents_norm,
        norm_mean=stats['mean'],
        norm_std=stats['std']
    )
    
    # Save metadata as CSV (easier to inspect in Excel/pandas)
    metadata.to_csv(output_dir / 'metadata.csv', index=False)
    
    # Save full stats dict
    with open(output_dir / 'stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"✓ Saved arrays to {filepath}")
    print(f"✓ Saved metadata to {output_dir / 'metadata.csv'}")
    print(f"✓ Saved stats to {output_dir / 'stats.pkl'}")


def load_processed_data(data_dir):
    """
    Load processed dataset from NPZ files.
    
    Args:
        data_dir: Directory containing dataset.npz and metadata.csv
    
    Returns:
        latents_normalized, metadata, stats
    """
    data_dir = Path(data_dir)
    
    # Load arrays
    data = np.load(data_dir / 'dataset.npz')
    latents_normalized = data['latents_normalized']
    
    # Load metadata
    metadata = pd.read_csv(data_dir / 'metadata.csv')
    
    # Load stats
    with open(data_dir / 'stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    
    print(f"Loaded {len(latents_normalized):,} samples from {data_dir}")
    return latents_normalized, metadata, stats