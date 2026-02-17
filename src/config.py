"""Configuration constants for protein-ligand SAE project"""

# Data paths
ROOT_DIR = "/mnt/beegfs/home/friesner/bgl2126/data/12_09_25/for_bridget_2025.12.03"
OUTPUT_DIR = "./processed_data"

# Exclusions
EXCLUSION_LIST = [
    'hsp90_2ccu',
    'chk1_3hxl'
]

# Model parameters
LATENT_DIM = 30

# Quality thresholds
RMSD_GOOD_THRESHOLD = 2.0  # Angstroms