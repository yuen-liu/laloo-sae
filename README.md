# Sparse Autoencoder for Protein-Ligand Docking Pose Analysis

A computational biology and machine learning project that uses Sparse Autoencoders (SAE) to identify latent features that correlate with successful protein-ligand docking poses. This research aims to improve VAE-diffusion docking pipelines by enabling early filtering of poor poses.

## 🧬 Scientific Motivation

### Problem Statement
Protein-ligand docking is a critical step in drug discovery, but current methods often generate many poor-quality poses that need to be filtered out. Traditional approaches rely on energy-based scoring functions, which may not capture the complex structural features that distinguish good poses from bad ones.

### Research Goal
We have ~12,735 latent vectors (each 30D) representing protein-ligand docking poses, with corresponding genetic algorithm (GA) rankings and RMSD (pose quality) values. Our goal is to:

1. **Train a Sparse Autoencoder** to learn meaningful representations of docking poses
2. **Identify sparse features** that correlate with successful poses (<2Å RMSD) vs failures (≥3Å RMSD)
3. **Enable early filtering** of poor poses in VAE-diffusion docking pipelines
4. **Provide interpretability** into what makes a docking pose successful

### Expected Impact
- **Efficiency**: Filter poor poses early, reducing computational cost
- **Insight**: Understand molecular features that predict pose quality
- **Guidance**: Inform future model development and feature engineering

## 🏗️ Project Structure

```
laloo-sae/
├── src/                    # Core Python modules
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # SparseAutoencoder implementation
│   ├── train.py           # Training pipeline
│   ├── analysis.py        # Interpretability analysis
│   └── utils.py           # Utility functions
├── configs/               # Configuration files
│   └── config.yaml        # Main configuration
├── notebooks/             # Jupyter notebooks
│   └── interpretability.ipynb  # Interactive analysis
├── data/                  # Data directory
├── models/                # Saved model checkpoints
├── results/               # Analysis results and plots
├── main.py                # Main entry point
├── pyproject.toml         # Python package configuration
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd laloo-sae

# Install dependencies
pip install -e .

# Or install manually
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml jupyter
```

### 2. Create Sample Data (for testing)

```bash
python main.py --create-sample-data
```

### 3. Train the Model

```bash
# Train with default configuration
python main.py --mode train

# Train with custom config
python main.py --config configs/config.yaml --mode train
```

### 4. Run Analysis

```bash
# Run full pipeline (train + analyze)
python main.py --mode full

# Run analysis only (requires trained model)
python main.py --mode analyze --model models/best_model.pt
```

### 5. Interactive Analysis

```bash
# Launch Jupyter notebook for detailed analysis
jupyter notebook notebooks/interpretability.ipynb
```

## 📊 Usage Examples

### Basic Training

```python
from src.data_loader import load_docking_data
from src.model import create_model
from src.train import train_model
from src.utils import get_device, load_config

# Load configuration
config = load_config("configs/config.yaml")

# Load data
train_loader, val_loader, test_loader, scaler = load_docking_data(
    config['data']['data_path'], config['data']
)

# Train model
device = get_device()
model, history = train_model(config, train_loader, val_loader, device)
```

### Feature Analysis

```python
from src.analysis import SAEAnalyzer

# Create analyzer
analyzer = SAEAnalyzer(model, device)

# Extract features
latents, hidden_features, ga_rankings, rmsd_values, quality_labels = analyzer.extract_features(test_loader)

# Analyze sparsity
sparsity_analysis = analyzer.analyze_sparsity(hidden_features)

# Compute correlations
correlation_analysis = analyzer.correlate_features_with_quality(
    hidden_features, ga_rankings, rmsd_values, quality_labels
)
```

### Model Configuration

```yaml
# configs/config.yaml
model:
  input_dim: 30          # Latent vector dimension
  hidden_dim: 64         # Hidden layer size
  sparsity_lambda: 0.01  # L1 regularization strength
  dropout_rate: 0.1      # Dropout for regularization

training:
  epochs: 100            # Training epochs
  learning_rate: 0.001   # Learning rate
  batch_size: 64         # Batch size
  early_stopping: true   # Enable early stopping
  patience: 20           # Early stopping patience
```

## 🔬 Key Features

### Sparse Autoencoder Architecture
- **Encoder**: 30D → 128D → 64D (with ReLU and dropout)
- **Decoder**: 64D → 128D → 30D (with ReLU and dropout)
- **Sparsity**: L1 regularization on hidden layer
- **Regularization**: Dropout and weight decay

### Analysis Capabilities
- **Sparsity Analysis**: Understand feature activation patterns
- **Correlation Analysis**: Identify features correlated with pose quality
- **Feature Importance**: Rank features by predictive power
- **Visualization**: PCA, t-SNE, and correlation plots
- **Comparison**: SAE vs PCA performance comparison

### Interpretability Tools
- **Feature Ranking**: Most important features for pose quality
- **Activation Patterns**: How features differ between good/poor poses
- **Biological Interpretation**: Connect features to molecular properties
- **Pipeline Integration**: Ready for VAE-diffusion model integration

## 📈 Expected Results

### Performance Metrics
- **Sparsity**: ~70-80% of features inactive (configurable)
- **Classification**: AUC > 0.8 for pose quality prediction
- **Reconstruction**: Low MSE for latent space reconstruction
- **Interpretability**: Clear feature-pose quality correlations

### Key Insights
- **Top Features**: 5-10 most predictive latent dimensions
- **Quality Thresholds**: Feature activation patterns for good vs poor poses
- **Filtering Strategy**: Early pose filtering recommendations
- **Biological Relevance**: Connection to molecular interaction patterns

## 🛠️ Customization

### Adding New Data Sources
The modular design allows easy integration of new data sources:

```python
# In data_loader.py
def load_custom_data(data_path, config):
    # Implement custom data loading
    # Return: train_loader, val_loader, test_loader, scaler
    pass
```

### Modifying Model Architecture
```python
# In model.py
class CustomSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        # Implement custom architecture
        pass
```

### Adding Analysis Methods
```python
# In analysis.py
class CustomAnalyzer(SAEAnalyzer):
    def custom_analysis(self, features, labels):
        # Implement custom analysis
        pass
```

## 📚 Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization
- **PyYAML**: Configuration management

### Optional Dependencies
- **Jupyter**: Interactive analysis
- **tqdm**: Progress bars
- **CUDA**: GPU acceleration (if available)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or collaborations, please contact:
- **Research Team**: research@example.com
- **Project Repository**: https://github.com/your-org/laloo-sae

## 🙏 Acknowledgments

- Computational biology research community
- PyTorch and scikit-learn developers
- Protein-ligand docking methodology researchers

---

**Note**: This is a research project. Results may vary depending on your specific data and use case. Please adapt the configuration and analysis methods as needed for your research objectives.
