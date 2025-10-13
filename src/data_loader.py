"""
Data loading and preprocessing module for protein-ligand docking pose analysis.

This module handles loading latent vectors, GA rankings, and RMSD values,
with proper normalization and train/validation/test splitting.
"""

import os
import pickle
import glob
import time
from datetime import datetime as dt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Union, Tuple, Optional, Dict, Any
import yaml

from schrodinger.structure import StructureReader


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


def load_structure_data(file_path: str) -> List:
    """
    Load structure data from .maegz file using Schrodinger StructureReader.
    
    Args:
        file_path: Path to the .maegz structure file
        
    Returns:
        List of structures from the file
    """
    # Load single structure
    st = StructureReader.read(file_path)
    # Load all structures in file
    st_list = [st for st in StructureReader(file_path)]
    return st_list


class DockingPoseDataset(Dataset):
    """
    PyTorch Dataset for protein-ligand docking pose data.
    
    Attributes:
        latents (torch.Tensor): Normalized latent vectors (N, 30)
        ga_rankings (torch.Tensor): Genetic algorithm rankings (N,)
        rmsd_values (torch.Tensor): RMSD values in Angstroms (N,)
        pose_quality (torch.Tensor): Binary quality labels (0: poor ≥3Å, 1: good <2Å)
    """
    
    def __init__(self, latents: np.ndarray, ga_rankings: np.ndarray, 
                 rmsd_values: np.ndarray, scaler: Optional[StandardScaler] = None):
        """
        Initialize the dataset.
        
        Args:
            latents: Raw latent vectors (N, 30)
            ga_rankings: GA rankings (N,)
            rmsd_values: RMSD values in Angstroms (N,)
            scaler: Optional pre-fitted scaler for latents
        """
        self.latents = torch.FloatTensor(latents)
        self.ga_rankings = torch.FloatTensor(ga_rankings)
        self.rmsd_values = torch.FloatTensor(rmsd_values)
        
        # Normalize latents if scaler provided
        if scaler is not None:
            self.latents = torch.FloatTensor(scaler.transform(latents))
        
        # Create binary quality labels: 1 for good poses (<2Å), 0 for poor poses (≥3Å)
        self.pose_quality = (self.rmsd_values < 2.0).float()
        
        # Filter out intermediate quality poses (2-3Å) for cleaner binary classification
        valid_mask = (self.rmsd_values < 2.0) | (self.rmsd_values >= 3.0)
        self.latents = self.latents[valid_mask]
        self.ga_rankings = self.ga_rankings[valid_mask]
        self.rmsd_values = self.rmsd_values[valid_mask]
        self.pose_quality = self.pose_quality[valid_mask]
    
    def __len__(self) -> int:
        return len(self.latents)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'latent': self.latents[idx],
            'ga_ranking': self.ga_rankings[idx],
            'rmsd': self.rmsd_values[idx],
            'quality': self.pose_quality[idx]
        }


def load_docking_data(data_path: str, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Load and preprocess docking pose data with train/val/test splitting.
    
    Args:
        data_path: Path to the data file (CSV or NPZ format)
        config: Configuration dictionary with data loading parameters
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Load data based on file extension
    if data_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(data_path)
        latents = df[config['latent_columns']].values  # Assuming columns like 'latent_0', 'latent_1', etc.
        ga_rankings = df[config['ga_ranking_column']].values
        rmsd_values = df[config['rmsd_column']].values
    elif data_path.endswith('.npz'):
        data = np.load(data_path)
        latents = data['latents']
        ga_rankings = data['ga_rankings']
        rmsd_values = data['rmsd_values']
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded {len(latents)} docking poses")
    print(f"Latent vector shape: {latents.shape}")
    print(f"RMSD range: {rmsd_values.min():.2f} - {rmsd_values.max():.2f} Å")
    print(f"Good poses (<2Å): {(rmsd_values < 2.0).sum()}")
    print(f"Poor poses (≥3Å): {(rmsd_values >= 3.0).sum()}")
    
    # Normalize latent vectors
    scaler = StandardScaler()
    latents_normalized = scaler.fit_transform(latents)
    
    # Split data
    train_size = config.get('train_size', 0.7)
    val_size = config.get('val_size', 0.15)
    test_size = config.get('test_size', 0.15)
    
    # First split: train vs (val + test)
    X_temp, X_test, y_temp, y_test, ga_temp, ga_test, rmsd_temp, rmsd_test = train_test_split(
        latents_normalized, latents_normalized, ga_rankings, rmsd_values,
        test_size=test_size, random_state=config.get('random_seed', 42)
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ga_train, ga_val, rmsd_train, rmsd_val = train_test_split(
        X_temp, y_temp, ga_temp, rmsd_temp,
        test_size=val_size_adjusted, random_state=config.get('random_seed', 42)
    )
    
    # Create datasets
    train_dataset = DockingPoseDataset(X_train, ga_train, rmsd_train)
    val_dataset = DockingPoseDataset(X_val, ga_val, rmsd_val)
    test_dataset = DockingPoseDataset(X_test, ga_test, rmsd_test)
    
    # Create data loaders
    batch_size = config.get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, scaler


def create_sample_data(output_path: str, n_samples: int = 12735, latent_dim: int = 30) -> None:
    """
    Create sample data for testing purposes.
    
    Args:
        output_path: Path to save the sample data
        n_samples: Number of samples to generate
        latent_dim: Dimension of latent vectors
    """
    np.random.seed(42)
    
    # Generate realistic latent vectors
    latents = np.random.randn(n_samples, latent_dim)
    
    # Generate GA rankings (lower is better, 1-1000 range)
    ga_rankings = np.random.randint(1, 1001, n_samples)
    
    # Generate RMSD values with realistic distribution
    # Good poses: 0.5-2.0 Å, Poor poses: 3.0-8.0 Å
    good_poses = np.random.uniform(0.5, 2.0, n_samples // 2)
    poor_poses = np.random.uniform(3.0, 8.0, n_samples - n_samples // 2)
    rmsd_values = np.concatenate([good_poses, poor_poses])
    np.random.shuffle(rmsd_values)
    
    # Save as NPZ file
    np.savez(output_path, 
             latents=latents, 
             ga_rankings=ga_rankings, 
             rmsd_values=rmsd_values)
    
    print(f"Sample data saved to {output_path}")
    print(f"Generated {n_samples} samples with {latent_dim}D latent vectors")


if __name__ == "__main__":
    # Load latent vector data using the pattern from example.ipynb
    files = sorted(glob.glob('pim1_4lmu_pim1_4bzo/pim1_4lmu_pim1_4bzo_optimization/*.pkl'))
    z_list, E_list, sr_list, lr_list, gen_list = load_latents(files)
    
    print(f"Loaded {len(z_list)} latent vectors")
    print(f"Latent vector shape: {z_list.shape}")
    print(f"Energy range: {E_list.min():.2f} - {E_list.max():.2f}")
    print(f"Site RMSD range: {sr_list.min():.2f} - {sr_list.max():.2f}")
    print(f"Ligand RMSD range: {lr_list.min():.2f} - {lr_list.max():.2f}")
    print(f"Generation range: {gen_list.min()} - {gen_list.max()}")
    
    # Example of loading structure data
    # file = 'pim1_4lmu_pim1_4bzo/pim1_4lmu_pim1_4bzo_optimization/pim1_4lmu_pim1_4bzo_opt_R0-ALL-out.maegz'
    # st_list = load_structure_data(file)
    # print(f"Loaded {len(st_list)} structures")
