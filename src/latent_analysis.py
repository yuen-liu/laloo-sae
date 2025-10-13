"""
Latent Analysis Module for protein-ligand docking pose data.

This module provides comprehensive analysis of latent vectors including
statistical analysis, visualization, correlation analysis, and quality assessment.
"""

import os
import pickle
import glob
import time
from datetime import datetime as dt
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Union, Tuple, Optional, Dict, Any
import json


def load_latents(files: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads latent vectors from .pkl files

    Returns:
        z_list: torch.Tensor; list of latent vectors
        E_list: np.ndarray; list of energies for each latent vector (same order)
        sr_list: np.ndarray; list of protein site rmsd for each latent vector (same order)
        lr_list: np.ndarray; list of ligand rmsd for each latent vector (same order)
        gen_list: np.ndarray; generation of GA each latent vector is from (same order)
    """
    z_list, E_list, sr_list, lr_list, gen_list = [], [], [], [], []
    for file in files:
        with open(file, 'rb') as f:
            ld = pickle.load(f)
        if 'site_rmsd' not in ld['scores']:
            continue
        if 'lig_rmsd' not in ld['scores']:
            continue
        z_list.append(ld['z'])
        E_list.append(ld['scores']['energy'])
        sr_list.append(ld['scores']['site_rmsd'])
        lr_list.append(ld['scores']['lig_rmsd'])
        gen_list.append(ld['curr_gen'])
    z_list, E_list, sr_list, lr_list, gen_list = \
        torch.stack(z_list), np.array(E_list), np.array(sr_list), \
        np.array(lr_list), np.array(gen_list)
    return z_list, E_list, sr_list, lr_list, gen_list


def analyze_latent_statistics(z_list: torch.Tensor, E_list: np.ndarray, sr_list: np.ndarray, 
                            lr_list: np.ndarray, gen_list: np.ndarray) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on latent vectors and associated metrics.
    
    Args:
        z_list: Latent vectors
        E_list: Energy values
        sr_list: Site RMSD values
        lr_list: Ligand RMSD values
        gen_list: Generation numbers
        
    Returns:
        Dictionary containing comprehensive statistics
    """
    # Convert to numpy for analysis
    z_np = z_list.numpy() if isinstance(z_list, torch.Tensor) else z_list
    
    analysis = {
        'dataset_info': {
            'n_samples': len(z_list),
            'latent_dim': z_np.shape[1],
            'n_generations': len(np.unique(gen_list)),
            'generation_range': [int(np.min(gen_list)), int(np.max(gen_list))]
        },
        'latent_statistics': {
            'mean': np.mean(z_np, axis=0).tolist(),
            'std': np.std(z_np, axis=0).tolist(),
            'min': np.min(z_np, axis=0).tolist(),
            'max': np.max(z_np, axis=0).tolist(),
            'median': np.median(z_np, axis=0).tolist(),
            'overall_mean': float(np.mean(z_np)),
            'overall_std': float(np.std(z_np)),
            'sparsity': float(np.mean(np.abs(z_np) < 0.1))  # Fraction of near-zero values
        },
        'energy_statistics': {
            'mean': float(np.mean(E_list)),
            'std': float(np.std(E_list)),
            'min': float(np.min(E_list)),
            'max': float(np.max(E_list)),
            'median': float(np.median(E_list))
        },
        'site_rmsd_statistics': {
            'mean': float(np.mean(sr_list)),
            'std': float(np.std(sr_list)),
            'min': float(np.min(sr_list)),
            'max': float(np.max(sr_list)),
            'median': float(np.median(sr_list))
        },
        'ligand_rmsd_statistics': {
            'mean': float(np.mean(lr_list)),
            'std': float(np.std(lr_list)),
            'min': float(np.min(lr_list)),
            'max': float(np.max(lr_list)),
            'median': float(np.median(lr_list))
        }
    }
    
    # Quality classification based on ligand RMSD
    good_poses = lr_list < 2.0
    poor_poses = lr_list >= 3.0
    intermediate_poses = (lr_list >= 2.0) & (lr_list < 3.0)
    
    analysis['quality_distribution'] = {
        'good_poses': int(np.sum(good_poses)),
        'poor_poses': int(np.sum(poor_poses)),
        'intermediate_poses': int(np.sum(intermediate_poses)),
        'good_percentage': float(np.mean(good_poses) * 100),
        'poor_percentage': float(np.mean(poor_poses) * 100),
        'intermediate_percentage': float(np.mean(intermediate_poses) * 100)
    }
    
    # Generation-wise analysis
    unique_gens = np.unique(gen_list)
    gen_stats = {}
    for gen in unique_gens:
        gen_mask = gen_list == gen
        gen_stats[f'gen_{int(gen)}'] = {
            'n_samples': int(np.sum(gen_mask)),
            'mean_energy': float(np.mean(E_list[gen_mask])),
            'mean_ligand_rmsd': float(np.mean(lr_list[gen_mask])),
            'good_poses': int(np.sum(good_poses[gen_mask])),
            'poor_poses': int(np.sum(poor_poses[gen_mask]))
        }
    analysis['generation_analysis'] = gen_stats
    
    return analysis


def analyze_latent_correlations(z_list: torch.Tensor, E_list: np.ndarray, sr_list: np.ndarray, 
                              lr_list: np.ndarray, gen_list: np.ndarray) -> Dict[str, Any]:
    """
    Analyze correlations between latent dimensions and quality metrics.
    
    Args:
        z_list: Latent vectors
        E_list: Energy values
        sr_list: Site RMSD values
        lr_list: Ligand RMSD values
        gen_list: Generation numbers
        
    Returns:
        Dictionary containing correlation analysis
    """
    # Convert to numpy
    z_np = z_list.numpy() if isinstance(z_list, torch.Tensor) else z_list
    
    # Create DataFrame for correlation analysis
    data_dict = {
        'energy': E_list,
        'site_rmsd': sr_list,
        'ligand_rmsd': lr_list,
        'generation': gen_list
    }
    
    # Add all latent dimensions
    for i in range(z_np.shape[1]):
        data_dict[f'latent_{i}'] = z_np[:, i]
    
    df = pd.DataFrame(data_dict)
    correlation_matrix = df.corr()
    
    # Find strongest correlations with ligand RMSD (main quality metric)
    ligand_rmsd_corr = correlation_matrix['ligand_rmsd'].abs().sort_values(ascending=False)
    
    # Find latent dimensions most correlated with quality
    latent_cols = [col for col in df.columns if col.startswith('latent_')]
    latent_quality_corr = correlation_matrix.loc[latent_cols, 'ligand_rmsd'].abs().sort_values(ascending=False)
    
    analysis = {
        'correlation_matrix': correlation_matrix.to_dict(),
        'ligand_rmsd_correlations': ligand_rmsd_corr.to_dict(),
        'latent_quality_correlations': latent_quality_corr.to_dict(),
        'key_correlations': {
            'energy_ligand_rmsd': float(correlation_matrix.loc['energy', 'ligand_rmsd']),
            'site_ligand_rmsd': float(correlation_matrix.loc['site_rmsd', 'ligand_rmsd']),
            'generation_ligand_rmsd': float(correlation_matrix.loc['generation', 'ligand_rmsd']),
            'top_latent_quality_corr': float(latent_quality_corr.iloc[0]) if len(latent_quality_corr) > 0 else 0.0
        }
    }
    
    return analysis


def visualize_latent_distributions(z_list: torch.Tensor, E_list: np.ndarray, sr_list: np.ndarray, 
                                 lr_list: np.ndarray, gen_list: np.ndarray, 
                                 save_dir: str = "results") -> None:
    """
    Create comprehensive visualizations of latent space and data distributions.
    
    Args:
        z_list: Latent vectors
        E_list: Energy values
        sr_list: Site RMSD values
        lr_list: Ligand RMSD values
        gen_list: Generation numbers
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Convert to numpy
    z_np = z_list.numpy() if isinstance(z_list, torch.Tensor) else z_list
    
    # 1. Data Distribution Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Energy distribution
    axes[0, 0].hist(E_list, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Energy Distribution')
    axes[0, 0].set_xlabel('Energy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(E_list), color='red', linestyle='--', label=f'Mean: {np.mean(E_list):.2f}')
    axes[0, 0].legend()
    
    # Site RMSD distribution
    axes[0, 1].hist(sr_list, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Site RMSD Distribution')
    axes[0, 1].set_xlabel('Site RMSD (Å)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(sr_list), color='red', linestyle='--', label=f'Mean: {np.mean(sr_list):.2f}')
    axes[0, 1].legend()
    
    # Ligand RMSD distribution
    axes[0, 2].hist(lr_list, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_title('Ligand RMSD Distribution')
    axes[0, 2].set_xlabel('Ligand RMSD (Å)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(2.0, color='red', linestyle='--', label='Good/Bad threshold')
    axes[0, 2].axvline(3.0, color='red', linestyle='--', label='Bad threshold')
    axes[0, 2].legend()
    
    # Generation distribution
    axes[1, 0].hist(gen_list, bins=len(np.unique(gen_list)), alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].set_title('Generation Distribution')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Frequency')
    
    # Energy vs Ligand RMSD scatter
    scatter = axes[1, 1].scatter(E_list, lr_list, alpha=0.6, c=gen_list, cmap='viridis')
    axes[1, 1].set_title('Energy vs Ligand RMSD')
    axes[1, 1].set_xlabel('Energy')
    axes[1, 1].set_ylabel('Ligand RMSD (Å)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Generation')
    
    # Site RMSD vs Ligand RMSD scatter
    scatter = axes[1, 2].scatter(sr_list, lr_list, alpha=0.6, c=gen_list, cmap='viridis')
    axes[1, 2].set_title('Site RMSD vs Ligand RMSD')
    axes[1, 2].set_xlabel('Site RMSD (Å)')
    axes[1, 2].set_ylabel('Ligand RMSD (Å)')
    plt.colorbar(scatter, ax=axes[1, 2], label='Generation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Latent Space Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Latent Space Analysis', fontsize=16, fontweight='bold')
    
    # PCA visualization
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)
    
    scatter = axes[0, 0].scatter(z_pca[:, 0], z_pca[:, 1], c=lr_list, cmap='viridis', alpha=0.6)
    axes[0, 0].set_title('Latent Space (PCA) - Colored by Ligand RMSD')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Ligand RMSD (Å)')
    
    # PCA colored by generation
    scatter = axes[0, 1].scatter(z_pca[:, 0], z_pca[:, 1], c=gen_list, cmap='plasma', alpha=0.6)
    axes[0, 1].set_title('Latent Space (PCA) - Colored by Generation')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Generation')
    
    # Latent dimension distributions (first 4 dimensions)
    for i in range(min(4, z_np.shape[1])):
        row = 1
        col = i
        axes[row, col].hist(z_np[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Latent Dimension {i}')
        axes[row, col].set_xlabel(f'Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].axvline(np.mean(z_np[:, i]), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(z_np[:, i]):.3f}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_space_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Create correlation matrix for key metrics and top latent dimensions
    key_metrics = ['energy', 'site_rmsd', 'ligand_rmsd', 'generation']
    top_latent_dims = [f'latent_{i}' for i in range(min(10, z_np.shape[1]))]
    
    corr_data = pd.DataFrame({
        'energy': E_list,
        'site_rmsd': sr_list,
        'ligand_rmsd': lr_list,
        'generation': gen_list
    })
    
    for i in range(min(10, z_np.shape[1])):
        corr_data[f'latent_{i}'] = z_np[:, i]
    
    correlation_matrix = corr_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix: Metrics and Top Latent Dimensions')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {save_dir}/")
    print(f"  - data_distributions.png")
    print(f"  - latent_space_analysis.png") 
    print(f"  - correlation_heatmap.png")


def analyze_latent_quality_relationship(z_list: torch.Tensor, lr_list: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the relationship between latent vectors and pose quality.
    
    Args:
        z_list: Latent vectors
        lr_list: Ligand RMSD values
        
    Returns:
        Dictionary containing quality relationship analysis
    """
    z_np = z_list.numpy() if isinstance(z_list, torch.Tensor) else z_list
    
    # Define quality categories
    good_poses = lr_list < 2.0
    poor_poses = lr_list >= 3.0
    intermediate_poses = (lr_list >= 2.0) & (lr_list < 3.0)
    
    analysis = {
        'quality_categories': {
            'good': {'mask': good_poses, 'count': int(np.sum(good_poses))},
            'poor': {'mask': poor_poses, 'count': int(np.sum(poor_poses))},
            'intermediate': {'mask': intermediate_poses, 'count': int(np.sum(intermediate_poses))}
        }
    }
    
    # Compare latent statistics across quality categories
    for quality, data in analysis['quality_categories'].items():
        if data['count'] > 0:
            mask = data['mask']
            analysis['quality_categories'][quality].update({
                'mean_latent': np.mean(z_np[mask], axis=0).tolist(),
                'std_latent': np.std(z_np[mask], axis=0).tolist(),
                'mean_rmsd': float(np.mean(lr_list[mask])),
                'std_rmsd': float(np.std(lr_list[mask]))
            })
    
    # Find latent dimensions that best separate good from poor poses
    if np.sum(good_poses) > 0 and np.sum(poor_poses) > 0:
        good_latents = z_np[good_poses]
        poor_latents = z_np[poor_poses]
        
        # Calculate separation score for each dimension
        separation_scores = []
        for i in range(z_np.shape[1]):
            good_mean = np.mean(good_latents[:, i])
            poor_mean = np.mean(poor_latents[:, i])
            good_std = np.std(good_latents[:, i])
            poor_std = np.std(poor_latents[:, i])
            
            # Separation score (difference in means relative to combined std)
            combined_std = np.sqrt((good_std**2 + poor_std**2) / 2)
            if combined_std > 0:
                separation_score = abs(good_mean - poor_mean) / combined_std
            else:
                separation_score = 0
            separation_scores.append(separation_score)
        
        # Find best separating dimensions
        best_dims = np.argsort(separation_scores)[-10:]  # Top 10
        analysis['best_separating_dimensions'] = {
            'dimensions': best_dims.tolist(),
            'scores': [separation_scores[i] for i in best_dims]
        }
    
    return analysis


def save_analysis_results(statistics: Dict[str, Any], correlations: Dict[str, Any], 
                         quality_analysis: Dict[str, Any], save_dir: str = "results") -> None:
    """
    Save all analysis results to JSON file.
    
    Args:
        statistics: Results from statistical analysis
        correlations: Results from correlation analysis
        quality_analysis: Results from quality relationship analysis
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'timestamp': dt.now().isoformat(),
        'statistics': statistics,
        'correlations': correlations,
        'quality_analysis': quality_analysis
    }
    
    with open(os.path.join(save_dir, 'latent_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis results saved to {save_dir}/latent_analysis_results.json")


def run_complete_latent_analysis(data_path: str = "pim1_4lmu_pim1_4bzo/pim1_4lmu_pim1_4bzo_optimization/*.pkl",
                                save_dir: str = "results") -> Dict[str, Any]:
    """
    Run complete latent analysis pipeline.
    
    Args:
        data_path: Path pattern for pickle files
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing all analysis results
    """
    print("="*60)
    print("LATENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading latent data...")
    files = sorted(glob.glob(data_path))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {data_path}")
    
    z_list, E_list, sr_list, lr_list, gen_list = load_latents(files)
    print(f"   Loaded {len(z_list)} samples with {z_list.shape[1]} latent dimensions")
    
    # Statistical analysis
    print("\n2. Performing statistical analysis...")
    statistics = analyze_latent_statistics(z_list, E_list, sr_list, lr_list, gen_list)
    
    print(f"   Dataset: {statistics['dataset_info']['n_samples']} samples, {statistics['dataset_info']['latent_dim']} dimensions")
    print(f"   Generations: {statistics['dataset_info']['generation_range'][0]} to {statistics['dataset_info']['generation_range'][1]}")
    print(f"   Quality distribution:")
    print(f"     - Good poses: {statistics['quality_distribution']['good_poses']} ({statistics['quality_distribution']['good_percentage']:.1f}%)")
    print(f"     - Poor poses: {statistics['quality_distribution']['poor_poses']} ({statistics['quality_distribution']['poor_percentage']:.1f}%)")
    print(f"     - Intermediate: {statistics['quality_distribution']['intermediate_poses']} ({statistics['quality_distribution']['intermediate_percentage']:.1f}%)")
    
    # Correlation analysis
    print("\n3. Analyzing correlations...")
    correlations = analyze_latent_correlations(z_list, E_list, sr_list, lr_list, gen_list)
    
    print(f"   Key correlations with ligand RMSD:")
    print(f"     - Energy: {correlations['key_correlations']['energy_ligand_rmsd']:.3f}")
    print(f"     - Site RMSD: {correlations['key_correlations']['site_ligand_rmsd']:.3f}")
    print(f"     - Generation: {correlations['key_correlations']['generation_ligand_rmsd']:.3f}")
    print(f"     - Best latent dimension: {correlations['key_correlations']['top_latent_quality_corr']:.3f}")
    
    # Quality relationship analysis
    print("\n4. Analyzing quality relationships...")
    quality_analysis = analyze_latent_quality_relationship(z_list, lr_list)
    
    if 'best_separating_dimensions' in quality_analysis:
        best_dims = quality_analysis['best_separating_dimensions']['dimensions'][:5]
        best_scores = quality_analysis['best_separating_dimensions']['scores'][:5]
        print(f"   Top 5 dimensions for separating good/poor poses:")
        for i, (dim, score) in enumerate(zip(best_dims, best_scores)):
            print(f"     {i+1}. Dimension {dim}: separation score {score:.3f}")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    visualize_latent_distributions(z_list, E_list, sr_list, lr_list, gen_list, save_dir)
    
    # Save results
    print("\n6. Saving results...")
    save_analysis_results(statistics, correlations, quality_analysis, save_dir)
    
    print(f"\nAnalysis complete! Results saved to '{save_dir}' directory.")
    
    return {
        'statistics': statistics,
        'correlations': correlations,
        'quality_analysis': quality_analysis
    }


if __name__ == "__main__":
    # Run complete analysis
    results = run_complete_latent_analysis()
