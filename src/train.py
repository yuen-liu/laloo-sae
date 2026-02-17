"""Training utilities for SAE"""
import torch
import os
import numpy as np


def train_sae(model, train_loader, val_loader, optimizer, device, max_epochs, 
              k, hidden_dim, run_id, model_dir, use_wandb=False, wandb_project=None):
    """
    Train one SAE instance.
    
    Args:
        model: TopKSAE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        device: Device to train on
        max_epochs: Number of epochs
        k: k value for this model
        hidden_dim: Hidden dimension
        run_id: Run ID for saving
        model_dir: Directory to save models
        use_wandb: Whether to use wandb logging
        wandb_project: Wandb project name
    
    Returns:
        model, best_val_loss
    """
    from model import loss_fn
    
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=f'topksae_k{k}_run{run_id}',
            config={
                'k': k,
                'hidden_dim': hidden_dim,
                'max_epochs': max_epochs,
                'run_id': run_id,
            }
        )
    
    # Track losses
    train_losses, val_losses = [], []
    train_mse_losses, train_auxk_losses = [], []
    val_mse_losses, val_auxk_losses = [], []
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    
    model.reset_usage_tracking()
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mse = 0.0
        epoch_train_auxk = 0.0
        n_train_batches = 0
        
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            recons, auxk, _ = model(x)
            mse_loss, auxk_loss = loss_fn(x, recons, auxk)
            total_loss = mse_loss + auxk_loss
            total_loss.backward()
            optimizer.step()
            model._tie_decoder_weights()
            
            epoch_train_loss += total_loss.item()
            epoch_train_mse += mse_loss.item()
            epoch_train_auxk += auxk_loss.item()
            n_train_batches += 1
        
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_train_mse = epoch_train_mse / n_train_batches
        avg_train_auxk = epoch_train_auxk / n_train_batches
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0
        epoch_val_auxk = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recons = model.forward_val(x)
                mse_loss, auxk_loss = loss_fn(x, recons, None)
                total_loss = mse_loss + auxk_loss
                
                epoch_val_loss += total_loss.item()
                epoch_val_mse += mse_loss.item()
                epoch_val_auxk += auxk_loss.item()
                n_val_batches += 1
        
        avg_val_loss = epoch_val_loss / n_val_batches
        avg_val_mse = epoch_val_mse / n_val_batches
        avg_val_auxk = epoch_val_auxk / n_val_batches
        
        # Store histories
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_mse_losses.append(avg_train_mse)
        train_auxk_losses.append(avg_train_auxk)
        val_mse_losses.append(avg_val_mse)
        val_auxk_losses.append(avg_val_auxk)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Get dead neuron count
        dead_mask = model.auxk_mask_fn()
        num_dead = dead_mask.sum().item()
        
        # Log metrics
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mse": avg_train_mse,
                "train_auxk": avg_train_auxk,
                "val_mse": avg_val_mse,
                "val_auxk": avg_val_auxk,
                "dead_neurons": num_dead/hidden_dim,
                "dead_neurons_pct": (num_dead/hidden_dim)*100,
            })
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epochs:
            print(f"k={k}, Run {run_id}, Epoch {epoch+1}/{max_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Dead: {num_dead}/{hidden_dim}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'topksae_k{k}_run{run_id}.pt')
    torch.save(model.state_dict(), model_path)
    
    history_path = os.path.join(model_dir, f"training_history_k{k}_run{run_id}.pkl")
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mse': train_mse_losses,
        'train_auxk': train_auxk_losses,
        'val_mse': val_mse_losses,
        'val_auxk': val_auxk_losses,
        'best_val_loss': best_val_loss,
        'k': k,
        'run_id': run_id
    }, history_path)
    
    print(f"Finished training k={k}, run {run_id}. Model saved to {model_path}")
    
    if use_wandb:
        wandb.log({
            "final_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1]
        })
        wandb.finish()
    
    return model, best_val_loss