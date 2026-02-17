"""PyTorch DataLoader for protein-ligand latent data"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LatentDataset(Dataset):
    """
    Dataset for latent vectors with optional metadata.
    """
    def __init__(self, latents, metadata=None, return_metadata=False):
        """
        Args:
            latents: numpy array of shape [N, latent_dim]
            metadata: pandas DataFrame with same length as latents (optional)
            return_metadata: if True, return (latent, metadata_dict) pairs
        """
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.metadata = metadata
        self.return_metadata = return_metadata
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent = self.latents[idx]
        
        if self.return_metadata and self.metadata is not None:
            meta = {
                'case_id': self.metadata.iloc[idx]['case_id'],
                'rmsd': float(self.metadata.iloc[idx]['rmsd']),
                'energy': float(self.metadata.iloc[idx]['energy']),
            }
            # Only include 'good_pose' if it exists in the metadata
            if 'good_pose' in self.metadata.columns:
                good_pose_val = self.metadata.iloc[idx]['good_pose']
                # Convert None to False, or keep the boolean value
                meta['good_pose'] = bool(good_pose_val) if good_pose_val is not None else False
            return latent, meta
        
        return latent


def create_dataloaders(
    latents,
    metadata,
    splits,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    return_metadata=False
):
    """
    Create train/val/test dataloaders from processed data.
    
    Args:
        latents: numpy array [N, latent_dim] - normalized latents
        metadata: pandas DataFrame with N rows
        splits: dict or npz file with 'train_idx', 'val_idx', 'test_idx'
        batch_size: batch size for training
        num_workers: number of workers for data loading
        pin_memory: whether to pin memory (faster GPU transfer)
        return_metadata: if True, dataloaders yield (latent, metadata) tuples
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load splits if npz file
    if isinstance(splits, str):
        splits = np.load(splits)
    
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']
    
    # Create datasets
    train_dataset = LatentDataset(
        latents[train_idx],
        metadata.iloc[train_idx] if metadata is not None else None,
        return_metadata=return_metadata
    )
    
    val_dataset = LatentDataset(
        latents[val_idx],
        metadata.iloc[val_idx] if metadata is not None else None,
        return_metadata=return_metadata
    )
    
    test_dataset = LatentDataset(
        latents[test_idx],
        metadata.iloc[test_idx] if metadata is not None else None,
        return_metadata=return_metadata
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Important for training
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def get_single_batch(dataloader):
    """Helper to get a single batch for testing"""
    return next(iter(dataloader))