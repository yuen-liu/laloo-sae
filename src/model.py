"""
Sparse Autoencoder model for protein-ligand docking pose analysis.

This module implements a Sparse Autoencoder with configurable sparsity penalty
to identify latent features that correlate with successful docking poses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1 regularization on the hidden layer.
    
    The model learns to reconstruct input latent vectors while encouraging
    sparsity in the hidden representation, which helps identify the most
    important features for pose quality prediction.
    
    Architecture:
        Input (30D) -> Encoder -> Hidden (configurable) -> Decoder -> Output (30D)
    """
    
    def __init__(self, input_dim: int = 30, hidden_dim: int = 64, 
                 sparsity_lambda: float = 0.01, dropout_rate: float = 0.1):
        """
        Initialize the Sparse Autoencoder.
        
        Args:
            input_dim: Dimension of input latent vectors (default: 30)
            hidden_dim: Dimension of hidden layer (default: 64)
            sparsity_lambda: L1 regularization strength for sparsity (default: 0.01)
            dropout_rate: Dropout rate for regularization (default: 0.1)
        """
        super(SparseAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Hidden representation of shape (batch_size, hidden_dim)
        """
        return self.encoder(x)
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to output.
        
        Args:
            h: Hidden representation of shape (batch_size, hidden_dim)
            
        Returns:
            Reconstructed output of shape (batch_size, input_dim)
        """
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed_output, hidden_representation)
        """
        hidden = self.encode(x)
        reconstructed = self.decode(hidden)
        return reconstructed, hidden
    
    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor, 
                    hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss including reconstruction and sparsity terms.
        
        Args:
            x: Original input tensor
            reconstructed: Reconstructed output tensor
            hidden: Hidden representation tensor
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # Sparsity loss (L1 regularization on hidden layer)
        sparsity_loss = torch.mean(torch.abs(hidden))
        
        # Total loss
        total_loss = reconstruction_loss + self.sparsity_lambda * sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def get_sparse_features(self, x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Extract sparse features by applying a threshold to the hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            threshold: Threshold for sparsity (features below this are set to 0)
            
        Returns:
            Sparse hidden representation
        """
        with torch.no_grad():
            hidden = self.encode(x)
            sparse_hidden = torch.where(torch.abs(hidden) > threshold, hidden, torch.zeros_like(hidden))
            return sparse_hidden
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance as the absolute values of hidden activations.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Feature importance scores of shape (batch_size, hidden_dim)
        """
        with torch.no_grad():
            hidden = self.encode(x)
            return torch.abs(hidden)


class PoseQualityPredictor(nn.Module):
    """
    Auxiliary classifier to predict pose quality from sparse features.
    
    This can be used to validate that the learned sparse features are
    meaningful for pose quality prediction.
    """
    
    def __init__(self, hidden_dim: int = 64, dropout_rate: float = 0.2):
        """
        Initialize the pose quality predictor.
        
        Args:
            hidden_dim: Dimension of input features (should match SAE hidden dim)
            dropout_rate: Dropout rate for regularization
        """
        super(PoseQualityPredictor, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict pose quality from sparse features.
        
        Args:
            features: Sparse features of shape (batch_size, hidden_dim)
            
        Returns:
            Quality predictions of shape (batch_size, 1)
        """
        return self.classifier(features)


def create_model(config: Dict[str, Any]) -> SparseAutoencoder:
    """
    Create a SparseAutoencoder model from configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Initialized SparseAutoencoder model
    """
    model = SparseAutoencoder(
        input_dim=config.get('input_dim', 30),
        hidden_dim=config.get('hidden_dim', 64),
        sparsity_lambda=config.get('sparsity_lambda', 0.01),
        dropout_rate=config.get('dropout_rate', 0.1)
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    config = {
        'input_dim': 30,
        'hidden_dim': 64,
        'sparsity_lambda': 0.01,
        'dropout_rate': 0.1
    }
    
    model = create_model(config)
    print(f"Model created with {count_parameters(model)} parameters")
    
    # Test forward pass
    x = torch.randn(32, 30)
    reconstructed, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Output shape: {reconstructed.shape}")
    
    # Test loss computation
    losses = model.compute_loss(x, reconstructed, hidden)
    print(f"Losses: {losses}")
