#!/usr/bin/env python3
"""
Train TopK SAEs across k values on protein-ligand docking latents.

Standalone script version of notebooks/sae_all.ipynb's training loop, meant
to be sbatch'd instead of babysat in a notebook kernel. Uses the shared
TopKSAE / train_sae implementations in src/ (not a notebook-local copy), so
fixes made there (e.g. b_pre initialization) apply here automatically.

Usage:
    python scripts/train_sae.py
    python scripts/train_sae.py --k-values 3 8 15 --n-runs 3
    python scripts/train_sae.py --model-dir models/07_30_26
"""
import argparse
import os
import sys
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))  # src/train.py imports `model` directly, not `src.model`

from src.config import OUTPUT_DIR
from src.data_processor import load_processed_data
from src.dataloader import create_dataloaders
from src.model import TopKSAE
from src.train import train_sae


def parse_args():
    parser = argparse.ArgumentParser(description="Train TopK SAEs across k values")
    parser.add_argument("--data-dir", default=OUTPUT_DIR,
                         help="Directory with dataset.npz/splits.npz (default: src/config.OUTPUT_DIR)")
    parser.add_argument("--model-dir", default=None,
                         help="Where to save models (default: models/<today's date, MM_DD_YY>)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 6, 8, 15, 20, 30])
    parser.add_argument("--hidden-dim", type=int, default=120)
    parser.add_argument("--input-dim", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--auxk", type=int, default=12)
    parser.add_argument("--dead-steps-threshold", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default="laloo-sae")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    args = parse_args()

    model_dir = args.model_dir or os.path.join("models", date.today().strftime("%m_%d_%y"))
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model dir: {model_dir}")

    print(f"\nLoading processed data from: {args.data_dir}")
    latents, metadata, stats = load_processed_data(args.data_dir)
    splits_path = os.path.join(args.data_dir, "splits.npz")

    train_loader, val_loader, test_loader = create_dataloaders(
        latents, metadata, splits_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    results = {k: [] for k in args.k_values}

    print(f"\nStarting training: {len(args.k_values)} k values x {args.n_runs} runs "
          f"= {len(args.k_values) * args.n_runs} total models")
    print("=" * 60)

    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"Training k={k}")
        print(f"Sparsity: {(1 - k / args.hidden_dim) * 100:.1f}% ({k}/{args.hidden_dim} active)")
        print(f"{'='*60}")

        for run_id in range(args.n_runs):
            seed = args.base_seed + k * 100 + run_id
            print(f"\n  Run {run_id + 1}/{args.n_runs} (seed={seed})")
            set_seed(seed)

            model = TopKSAE(
                input_dim=args.input_dim, hidden_dim=args.hidden_dim, k=k,
                auxk=args.auxk, batch_size=args.batch_size,
                dead_steps_threshold=args.dead_steps_threshold,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            try:
                _, best_val_loss = train_sae(
                    model, train_loader, val_loader, optimizer, device,
                    max_epochs=args.max_epochs, k=k, hidden_dim=args.hidden_dim,
                    run_id=run_id, model_dir=model_dir,
                    use_wandb=not args.no_wandb, wandb_project=args.wandb_project,
                )
                results[k].append({
                    "run_id": run_id,
                    "seed": seed,
                    "best_val_loss": best_val_loss,
                    "model_path": os.path.join(model_dir, f"topksae_k{k}_run{run_id}.pt"),
                })
            except Exception as e:
                print(f"ERROR in k={k}, run {run_id}: {e}")
                traceback.print_exc()
                continue

    summary_path = os.path.join(model_dir, "training_summary.pkl")
    torch.save(results, summary_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"{'k':<4} {'Sparsity':<12} {'Runs':<6} {'Val Loss (mean ± std)'}")
    print("-" * 60)
    for k in args.k_values:
        if results[k]:
            val_losses = [r["best_val_loss"] for r in results[k]]
            sparsity_pct = (1 - k / args.hidden_dim) * 100
            print(f"{k:<4} {sparsity_pct:>5.1f}%       {len(results[k]):<6} "
                  f"{np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        else:
            print(f"{k:<4} {'N/A':<12} {0:<6} No successful runs")

    print(f"\nAll models saved to: {model_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
