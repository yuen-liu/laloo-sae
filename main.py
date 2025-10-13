#!/usr/bin/env python3
"""
Main entry point for the Sparse Autoencoder protein-ligand docking pose analysis.

This script orchestrates the complete pipeline:
1. Load and preprocess data
2. Train the Sparse Autoencoder
3. Evaluate model performance
4. Run interpretability analysis
5. Generate comprehensive reports

Usage:
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --mode train
    python main.py --config configs/config.yaml --mode analyze --model models/best_model.pt
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import load_docking_data, create_sample_data
from model import create_model
from train import train_model
from analysis import run_interpretability_analysis
from utils import set_seed, get_device, load_config, create_results_directory, print_model_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sparse Autoencoder for Protein-Ligand Docking Pose Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "analyze", "full"],
        default="full",
        help="Execution mode: train, analyze, or full pipeline"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model for analysis mode"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (overrides config)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data for testing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def setup_environment(config: dict, args) -> tuple:
    """
    Setup the training environment.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Tuple of (device, results_dir)
    """
    # Set random seed
    set_seed(config.get('random_seed', 42))
    
    # Get device
    device = get_device()
    
    # Create results directory
    if args.output:
        results_dir = args.output
    else:
        results_dir = create_results_directory(config.get('analysis', {}).get('results_dir', 'results'))
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config to results directory
    config_path = os.path.join(results_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Results will be saved to: {results_dir}")
    print(f"Using device: {device}")
    
    return device, results_dir


def train_pipeline(config: dict, device: torch.device, results_dir: str, args) -> str:
    """
    Execute the training pipeline.
    
    Args:
        config: Configuration dictionary
        device: Device to train on
        results_dir: Directory to save results
        args: Command line arguments
        
    Returns:
        Path to the best trained model
    """
    print("\n" + "="*60)
    print("TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    data_config = config['data']
    if args.data:
        data_config['data_path'] = args.data
    
    print(f"Loading data from: {data_config['data_path']}")
    
    # Check if data file exists, create sample data if not
    if not os.path.exists(data_config['data_path']):
        print(f"Data file not found. Creating sample data...")
        create_sample_data(data_config['data_path'])
    
    train_loader, val_loader, test_loader, scaler = load_docking_data(
        data_config['data_path'], data_config
    )
    
    # Create model
    model_config = config['model']
    model = create_model(model_config)
    print_model_summary(model)
    
    # Training configuration
    train_config = {**model_config, **config['training']}
    train_config['save_dir'] = os.path.join(results_dir, 'models')
    
    # Train model
    print("\nStarting training...")
    trained_model, history = train_model(
        train_config, train_loader, val_loader, device
    )
    
    # Save final model
    model_path = os.path.join(train_config['save_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': train_config,
        'history': history
    }, model_path)
    
    print(f"\nTraining completed! Model saved to: {model_path}")
    
    return model_path


def analyze_pipeline(config: dict, device: torch.device, results_dir: str, 
                    model_path: str, args) -> dict:
    """
    Execute the analysis pipeline.
    
    Args:
        config: Configuration dictionary
        device: Device to run analysis on
        results_dir: Directory to save results
        model_path: Path to trained model
        args: Command line arguments
        
    Returns:
        Analysis results dictionary
    """
    print("\n" + "="*60)
    print("ANALYSIS PIPELINE")
    print("="*60)
    
    # Load data
    data_config = config['data']
    if args.data:
        data_config['data_path'] = args.data
    
    print(f"Loading data from: {data_config['data_path']}")
    train_loader, val_loader, test_loader, scaler = load_docking_data(
        data_config['data_path'], data_config
    )
    
    # Run analysis
    analysis_config = config.get('analysis', {})
    analysis_results = run_interpretability_analysis(
        model_path, test_loader, config['model'], 
        os.path.join(results_dir, 'analysis')
    )
    
    print(f"\nAnalysis completed! Results saved to: {os.path.join(results_dir, 'analysis')}")
    
    return analysis_results


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.verbose:
        print("Configuration loaded:")
        print(yaml.dump(config, default_flow_style=False, indent=2))
    
    # Setup environment
    device, results_dir = setup_environment(config, args)
    
    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample data...")
        create_sample_data("data/sample_docking_data.npz")
        print("Sample data created at: data/sample_docking_data.npz")
        return
    
    model_path = None
    
    # Execute based on mode
    if args.mode in ["train", "full"]:
        model_path = train_pipeline(config, device, results_dir, args)
    
    if args.mode in ["analyze", "full"]:
        if model_path is None:
            if args.model:
                model_path = args.model
            else:
                # Look for best model in results directory
                model_dir = os.path.join(results_dir, 'models')
                if os.path.exists(model_dir):
                    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model')]
                    if model_files:
                        model_path = os.path.join(model_dir, model_files[0])
                    else:
                        model_files = [f for f in os.listdir(model_dir) if f == 'final_model.pt']
                        if model_files:
                            model_path = os.path.join(model_dir, model_files[0])
                
                if model_path is None:
                    print("Error: No trained model found. Please train a model first or specify --model path")
                    return
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
        
        analyze_pipeline(config, device, results_dir, model_path, args)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
