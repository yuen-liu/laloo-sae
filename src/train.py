"""
Training module for the Sparse Autoencoder.

This module handles the training loop, loss tracking, validation,
and model checkpointing for the protein-ligand docking pose analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

from .model import SparseAutoencoder, PoseQualityPredictor
from .utils import set_seed, save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer class for the Sparse Autoencoder.
    
    Handles training loop, validation, loss tracking, and model checkpointing.
    """
    
    def __init__(self, model: SparseAutoencoder, config: Dict[str, Any], 
                 device: torch.device, save_dir: str = "models"):
        """
        Initialize the trainer.
        
        Args:
            model: SparseAutoencoder model to train
            config: Configuration dictionary
            device: Device to train on (cuda/cpu)
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('patience', 10),
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_reconstruction_losses = []
        self.val_reconstruction_losses = []
        self.train_sparsity_losses = []
        self.val_sparsity_losses = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing average losses for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            latents = batch['latent'].to(self.device)
            
            # Forward pass
            reconstructed, hidden = self.model(latents)
            
            # Compute loss
            losses = self.model.compute_loss(latents, reconstructed, hidden)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_reconstruction_loss += losses['reconstruction_loss'].item()
            total_sparsity_loss += losses['sparsity_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Recon': f"{losses['reconstruction_loss'].item():.4f}",
                'Sparse': f"{losses['sparsity_loss'].item():.4f}"
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_reconstruction_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing average validation losses
        """
        self.model.eval()
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                latents = batch['latent'].to(self.device)
                
                # Forward pass
                reconstructed, hidden = self.model(latents)
                
                # Compute loss
                losses = self.model.compute_loss(latents, reconstructed, hidden)
                
                # Accumulate losses
                total_loss += losses['total_loss'].item()
                total_reconstruction_loss += losses['reconstruction_loss'].item()
                total_sparsity_loss += losses['sparsity_loss'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_reconstruction_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            
        Returns:
            Dictionary containing training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Store history
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            self.train_reconstruction_losses.append(train_metrics['reconstruction_loss'])
            self.val_reconstruction_losses.append(val_metrics['reconstruction_loss'])
            self.train_sparsity_losses.append(train_metrics['sparsity_loss'])
            self.val_sparsity_losses.append(val_metrics['sparsity_loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Recon: {train_metrics['reconstruction_loss']:.4f}, "
                  f"Sparse: {train_metrics['sparsity_loss']:.4f})")
            print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(Recon: {val_metrics['reconstruction_loss']:.4f}, "
                  f"Sparse: {val_metrics['sparsity_loss']:.4f})")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.epochs_without_improvement = 0
                self.save_model(f"best_model_epoch_{epoch+1}.pt")
                print("New best model saved!")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if (self.config.get('early_stopping', False) and 
                self.epochs_without_improvement >= self.config.get('patience', 20)):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save final model
        self.save_model("final_model.pt")
        
        # Save training history
        self.save_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_reconstruction_losses': self.train_reconstruction_losses,
            'val_reconstruction_losses': self.val_reconstruction_losses,
            'train_sparsity_losses': self.train_sparsity_losses,
            'val_sparsity_losses': self.val_sparsity_losses
        }
    
    def save_model(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def save_training_history(self) -> None:
        """Save training history as JSON and plots."""
        # Save as JSON
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_reconstruction_losses': self.train_reconstruction_losses,
            'val_reconstruction_losses': self.val_reconstruction_losses,
            'train_sparsity_losses': self.train_sparsity_losses,
            'val_sparsity_losses': self.val_sparsity_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create plots
        self.plot_training_history()
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Validation', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(self.train_reconstruction_losses, label='Train', alpha=0.8)
        axes[0, 1].plot(self.val_reconstruction_losses, label='Validation', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sparsity loss
        axes[1, 0].plot(self.train_sparsity_losses, label='Train', alpha=0.8)
        axes[1, 0].plot(self.val_sparsity_losses, label='Validation', alpha=0.8)
        axes[1, 0].set_title('Sparsity Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lr_history, label='Learning Rate', alpha=0.8)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {self.save_dir}/training_history.png")


def train_model(config: Dict[str, Any], train_loader: DataLoader, 
                val_loader: DataLoader, device: torch.device) -> Tuple[SparseAutoencoder, Dict[str, List[float]]]:
    """
    Train a SparseAutoencoder model.
    
    Args:
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Set random seed
    set_seed(config.get('random_seed', 42))
    
    # Create model
    from .model import create_model
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Train model
    history = trainer.train(train_loader, val_loader, config.get('epochs', 100))
    
    return model, history


if __name__ == "__main__":
    # Test training with sample data
    from .data_loader import create_sample_data, load_docking_data
    
    # Create sample data
    create_sample_data("data/sample_docking_data.npz", n_samples=1000)
    
    # Load data
    config = {
        'batch_size': 32,
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_seed': 42
    }
    
    train_loader, val_loader, test_loader, scaler = load_docking_data(
        "data/sample_docking_data.npz", config
    )
    
    # Training config
    train_config = {
        'input_dim': 30,
        'hidden_dim': 64,
        'sparsity_lambda': 0.01,
        'learning_rate': 1e-3,
        'epochs': 10,
        'random_seed': 42
    }
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_model(train_config, train_loader, val_loader, device)
    
    print("Training completed!")
