"""
Sparse Autoencoder for Protein-Ligand Docking Pose Analysis

A computational biology and machine learning project that uses Sparse Autoencoders
to identify latent features that correlate with successful protein-ligand docking poses.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .data_loader import load_pickle_data
from .model import TopKSAE
from .train import train_sae
# from .analysis import SAEAnalyzer, run_interpretability_analysis
# from .utils import set_seed, get_device, load_config, save_config

__all__ = [
    "load_pickle_data",
    "TopKSAE",
    "PoseQualityPredictor",
    "create_model",
    "train_model",
    "Trainer",
    "SAEAnalyzer",
    "run_interpretability_analysis",
    "set_seed",
    "get_device",
    "load_config",
    "save_config"
]
