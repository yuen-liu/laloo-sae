"""
Analysis module for interpreting Sparse Autoencoder results.

This module provides functions for visualizing latent-space structure,
correlating sparse features with RMSD/GA rank, computing feature importance,
and running PCA for comparison with SAE results.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Tuple, Optional
import os
from tqdm import tqdm

from .model import SparseAutoencoder, PoseQualityPredictor
from .utils import plot_feature_importance, plot_correlation_heatmap, save_results


class SAEAnalyzer:
    """
    Analyzer class for interpreting Sparse Autoencoder results.
    
    Provides methods for feature analysis, visualization, and comparison
    with traditional dimensionality reduction techniques.
    """
    
    def __init__(self, model: SparseAutoencoder, device: torch.device):
        """
        Initialize the analyzer.
        
        Args:
            model: Trained SparseAutoencoder model
            device: Device to run analysis on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def extract_features(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and labels from data loader.
        
        Args:
            data_loader: PyTorch data loader
            
        Returns:
            Tuple of (latents, hidden_features, ga_rankings, rmsd_values, quality_labels)
        """
        latents = []
        hidden_features = []
        ga_rankings = []
        rmsd_values = []
        quality_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                latent = batch['latent'].to(self.device)
                ga_ranking = batch['ga_ranking'].to(self.device)
                rmsd = batch['rmsd'].to(self.device)
                quality = batch['quality'].to(self.device)
                
                # Get hidden representation
                hidden = self.model.encode(latent)
                
                latents.append(latent.cpu().numpy())
                hidden_features.append(hidden.cpu().numpy())
                ga_rankings.append(ga_ranking.cpu().numpy())
                rmsd_values.append(rmsd.cpu().numpy())
                quality_labels.append(quality.cpu().numpy())
        
        return (np.vstack(latents), np.vstack(hidden_features), 
                np.concatenate(ga_rankings), np.concatenate(rmsd_values), 
                np.concatenate(quality_labels))
    
    def compute_feature_importance(self, hidden_features: np.ndarray, 
                                 quality_labels: np.ndarray) -> np.ndarray:
        """
        Compute feature importance using Random Forest.
        
        Args:
            hidden_features: Hidden layer activations
            quality_labels: Binary quality labels
            
        Returns:
            Feature importance scores
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(hidden_features, quality_labels)
        return rf.feature_importances_
    
    def analyze_sparsity(self, hidden_features: np.ndarray, 
                        threshold: float = 0.1) -> Dict[str, Any]:
        """
        Analyze sparsity patterns in hidden features.
        
        Args:
            hidden_features: Hidden layer activations
            threshold: Threshold for considering a feature as active
            
        Returns:
            Dictionary containing sparsity analysis results
        """
        # Compute sparsity metrics
        active_features = np.abs(hidden_features) > threshold
        sparsity_per_sample = 1 - np.mean(active_features, axis=1)
        sparsity_per_feature = 1 - np.mean(active_features, axis=0)
        
        # Find most/least sparse features
        most_sparse_features = np.argsort(sparsity_per_feature)[:10]
        least_sparse_features = np.argsort(sparsity_per_feature)[-10:]
        
        return {
            'sparsity_per_sample': sparsity_per_sample,
            'sparsity_per_feature': sparsity_per_feature,
            'overall_sparsity': np.mean(sparsity_per_sample),
            'most_sparse_features': most_sparse_features,
            'least_sparse_features': least_sparse_features,
            'active_features': active_features
        }
    
    def correlate_features_with_quality(self, hidden_features: np.ndarray,
                                      ga_rankings: np.ndarray, 
                                      rmsd_values: np.ndarray,
                                      quality_labels: np.ndarray) -> Dict[str, Any]:
        """
        Correlate sparse features with pose quality metrics.
        
        Args:
            hidden_features: Hidden layer activations
            ga_rankings: GA ranking values
            rmsd_values: RMSD values
            quality_labels: Binary quality labels
            
        Returns:
            Dictionary containing correlation analysis results
        """
        # Compute correlations
        feature_quality_corr = np.corrcoef(hidden_features.T, quality_labels)[:-1, -1]
        feature_ga_corr = np.corrcoef(hidden_features.T, ga_rankings)[:-1, -1]
        feature_rmsd_corr = np.corrcoef(hidden_features.T, rmsd_values)[:-1, -1]
        
        # Find most correlated features
        top_quality_features = np.argsort(np.abs(feature_quality_corr))[-10:]
        top_ga_features = np.argsort(np.abs(feature_ga_corr))[-10:]
        top_rmsd_features = np.argsort(np.abs(feature_rmsd_corr))[-10:]
        
        return {
            'feature_quality_corr': feature_quality_corr,
            'feature_ga_corr': feature_ga_corr,
            'feature_rmsd_corr': feature_rmsd_corr,
            'top_quality_features': top_quality_features,
            'top_ga_features': top_ga_features,
            'top_rmsd_features': top_rmsd_features
        }
    
    def visualize_latent_space(self, latents: np.ndarray, hidden_features: np.ndarray,
                             quality_labels: np.ndarray, save_dir: str = "results") -> None:
        """
        Visualize latent space using PCA and t-SNE.
        
        Args:
            latents: Original latent vectors
            hidden_features: Hidden layer activations
            quality_labels: Binary quality labels
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # PCA on original latents
        pca_original = PCA(n_components=2)
        latents_pca = pca_original.fit_transform(latents)
        
        # PCA on hidden features
        pca_hidden = PCA(n_components=2)
        hidden_pca = pca_hidden.fit_transform(hidden_features)
        
        # t-SNE on hidden features (sample for speed)
        n_samples = min(5000, len(hidden_features))
        indices = np.random.choice(len(hidden_features), n_samples, replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        hidden_tsne = tsne.fit_transform(hidden_features[indices])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original latents PCA
        scatter = axes[0, 0].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                   c=quality_labels, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('Original Latents (PCA)')
        axes[0, 0].set_xlabel(f'PC1 ({pca_original.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca_original.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Pose Quality')
        
        # Hidden features PCA
        scatter = axes[0, 1].scatter(hidden_pca[:, 0], hidden_pca[:, 1], 
                                   c=quality_labels, cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Hidden Features (PCA)')
        axes[0, 1].set_xlabel(f'PC1 ({pca_hidden.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca_hidden.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Pose Quality')
        
        # Hidden features t-SNE
        scatter = axes[1, 0].scatter(hidden_tsne[:, 0], hidden_tsne[:, 1], 
                                   c=quality_labels[indices], cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Hidden Features (t-SNE)')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[1, 0], label='Pose Quality')
        
        # Feature importance
        feature_importance = self.compute_feature_importance(hidden_features, quality_labels)
        top_features = np.argsort(feature_importance)[-20:]
        axes[1, 1].barh(range(len(top_features)), feature_importance[top_features])
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_features])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 20 Feature Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Latent space visualization saved to {save_dir}/latent_space_visualization.png")
    
    def compare_with_pca(self, latents: np.ndarray, quality_labels: np.ndarray,
                        n_components: int = 64) -> Dict[str, Any]:
        """
        Compare SAE performance with PCA.
        
        Args:
            latents: Original latent vectors
            quality_labels: Binary quality labels
            n_components: Number of PCA components
            
        Returns:
            Dictionary containing comparison results
        """
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(latents)
        
        # Train classifiers on both representations
        # SAE features (need to extract from model)
        with torch.no_grad():
            latents_tensor = torch.FloatTensor(latents).to(self.device)
            sae_features = self.model.encode(latents_tensor).cpu().numpy()
        
        # Train Random Forest on both
        rf_sae = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
        
        rf_sae.fit(sae_features, quality_labels)
        rf_pca.fit(pca_features, quality_labels)
        
        # Get predictions
        sae_pred = rf_sae.predict_proba(sae_features)[:, 1]
        pca_pred = rf_pca.predict_proba(pca_features)[:, 1]
        
        # Compute AUC scores
        sae_auc = roc_auc_score(quality_labels, sae_pred)
        pca_auc = roc_auc_score(quality_labels, pca_pred)
        
        return {
            'sae_auc': sae_auc,
            'pca_auc': pca_auc,
            'sae_features': sae_features,
            'pca_features': pca_features,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'sae_importance': rf_sae.feature_importances_,
            'pca_importance': rf_pca.feature_importances_
        }
    
    def generate_report(self, data_loader, save_dir: str = "results") -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            data_loader: Data loader for analysis
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Extracting features...")
        latents, hidden_features, ga_rankings, rmsd_values, quality_labels = self.extract_features(data_loader)
        
        print("Analyzing sparsity...")
        sparsity_analysis = self.analyze_sparsity(hidden_features)
        
        print("Computing correlations...")
        correlation_analysis = self.correlate_features_with_quality(
            hidden_features, ga_rankings, rmsd_values, quality_labels
        )
        
        print("Comparing with PCA...")
        pca_comparison = self.compare_with_pca(latents, quality_labels)
        
        print("Generating visualizations...")
        self.visualize_latent_space(latents, hidden_features, quality_labels, save_dir)
        
        # Create summary report
        report = {
            'sparsity_analysis': sparsity_analysis,
            'correlation_analysis': correlation_analysis,
            'pca_comparison': pca_comparison,
            'data_summary': {
                'n_samples': len(latents),
                'n_features': hidden_features.shape[1],
                'good_poses': np.sum(quality_labels),
                'poor_poses': len(quality_labels) - np.sum(quality_labels),
                'mean_rmsd': np.mean(rmsd_values),
                'std_rmsd': np.std(rmsd_values)
            }
        }
        
        # Save report
        save_results(report, os.path.join(save_dir, 'analysis_report.json'))
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total samples: {report['data_summary']['n_samples']}")
        print(f"Good poses (<2Å): {report['data_summary']['good_poses']}")
        print(f"Poor poses (≥3Å): {report['data_summary']['poor_poses']}")
        print(f"Mean RMSD: {report['data_summary']['mean_rmsd']:.2f} Å")
        print(f"Overall sparsity: {sparsity_analysis['overall_sparsity']:.3f}")
        print(f"SAE AUC: {pca_comparison['sae_auc']:.3f}")
        print(f"PCA AUC: {pca_comparison['pca_auc']:.3f}")
        print("="*50)
        
        return report


def run_interpretability_analysis(model_path: str, data_loader, config: Dict[str, Any],
                                save_dir: str = "results") -> Dict[str, Any]:
    """
    Run complete interpretability analysis.
    
    Args:
        model_path: Path to trained model checkpoint
        data_loader: Data loader for analysis
        config: Configuration dictionary
        save_dir: Directory to save results
        
    Returns:
        Analysis results dictionary
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    from .model import create_model
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create analyzer
    analyzer = SAEAnalyzer(model, device)
    
    # Run analysis
    results = analyzer.generate_report(data_loader, save_dir)
    
    return results


if __name__ == "__main__":
    # Test analysis functions
    print("Testing analysis functions...")
    
    # Create dummy data
    latents = np.random.randn(1000, 30)
    quality_labels = np.random.randint(0, 2, 1000)
    
    # Test PCA comparison
    from .model import create_model
    model = create_model({'input_dim': 30, 'hidden_dim': 64})
    device = torch.device('cpu')
    analyzer = SAEAnalyzer(model, device)
    
    comparison = analyzer.compare_with_pca(latents, quality_labels)
    print(f"PCA comparison completed: SAE AUC = {comparison['sae_auc']:.3f}, PCA AUC = {comparison['pca_auc']:.3f}")
    
    print("Analysis functions tested successfully!")
