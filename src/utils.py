"""
Utility functions for the protein-ligand docking pose analysis project.

This module contains helper functions for seed setting, normalization,
checkpoint saving/loading, and other common utilities.
"""

import torch
import numpy as np
import random
import os
import json
import yaml
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, config: Dict[str, Any],
                   filepath: str, **kwargs) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        config: Configuration dictionary
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'Unknown')}")
    
    return checkpoint


def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    
    elif method == 'robust':
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        normalized_data = (data - median) / (mad + 1e-8)
        params = {'median': median, 'mad': mad}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data, params


def denormalize_data(normalized_data: np.ndarray, params: Dict[str, Any], 
                    method: str = 'standard') -> np.ndarray:
    """
    Denormalize data using stored parameters.
    
    Args:
        normalized_data: Normalized data array
        params: Normalization parameters
        method: Normalization method used
        
    Returns:
        Denormalized data array
    """
    if method == 'standard':
        return normalized_data * params['std'] + params['mean']
    
    elif method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    
    elif method == 'robust':
        return normalized_data * params['mad'] + params['median']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_correlation_matrix(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between features and labels.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        
    Returns:
        Correlation matrix
    """
    # Combine features and labels
    data = np.column_stack([features, labels])
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data.T)
    
    return corr_matrix


def plot_feature_importance(feature_importance: np.ndarray, feature_names: Optional[list] = None,
                          top_k: int = 20, save_path: Optional[str] = None) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_importance: Feature importance scores
        feature_names: Optional feature names
        top_k: Number of top features to plot
        save_path: Optional path to save plot
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
    
    # Get top-k features
    top_indices = np.argsort(feature_importance)[-top_k:]
    top_importance = feature_importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), top_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_k} Feature Importance Scores')
    plt.grid(True, alpha=0.3)
    
    # Color bars by importance
    colors = plt.cm.viridis(top_importance / top_importance.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(corr_matrix: np.ndarray, feature_names: Optional[list] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix
        feature_names: Optional feature names
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    if feature_names:
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.yticks(range(len(feature_names)), feature_names, rotation=0)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()


def create_results_directory(base_dir: str = "results") -> str:
    """
    Create a timestamped results directory.
    
    Args:
        base_dir: Base directory for results
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, np.integer):
            json_results[key] = int(value)
        elif isinstance(value, np.floating):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print layer information
    print("\nLayer details:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                print(f"  {name}: {module} - {num_params:,} parameters")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print("Seed set successfully")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test data normalization
    data = np.random.randn(100, 10)
    normalized_data, params = normalize_data(data, method='standard')
    denormalized_data = denormalize_data(normalized_data, params, method='standard')
    print(f"Normalization test: {np.allclose(data, denormalized_data, atol=1e-6)}")
    
    # Test results directory creation
    results_dir = create_results_directory()
    print(f"Results directory created: {results_dir}")
    
    print("All utility functions tested successfully!")
