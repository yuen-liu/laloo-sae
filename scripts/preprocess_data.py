#!/usr/bin/env python3
"""
Preprocess raw BRAID docking pickles into normalized latents + splits.

Standalone script version of notebooks/sae_all.ipynb's cell 4 (load +
normalize) / cell 9 (case split). Those are now one atomic step in
src.data_processor.process_all_cases: cases are split into train/val/test
BEFORE normalization stats are computed, so val/test poses never leak into
the mean/std used to normalize latents.

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --root-dir /path/to/raw --output-dir processed_data
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import ROOT_DIR, OUTPUT_DIR, EXCLUSION_LIST, LATENT_DIM
from src.data_processor import process_all_cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw docking poses into normalized latents + splits"
    )
    parser.add_argument("--root-dir", default=ROOT_DIR,
                         help="Directory with per-case *_optimization/*_initialization folders")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                         help="Where to write dataset.npz/splits.npz/metadata.csv/stats.pkl/case_info.pkl")
    parser.add_argument("--exclude", nargs="*", default=EXCLUSION_LIST,
                         help="Case-id substrings to exclude")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    return parser.parse_args()


def main():
    args = parse_args()
    process_all_cases(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        exclusion_list=args.exclude,
        latent_dim=args.latent_dim,
    )


if __name__ == "__main__":
    main()
