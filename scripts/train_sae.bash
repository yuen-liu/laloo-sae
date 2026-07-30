#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -J train_sae
#SBATCH -o train_sae_%j.out

# Load schrodinger and activate environment
module unload schrodinger; export SCHRODINGER=/cm/shared/apps/schrodinger/builds/NB/2025-4/build-050
source /mnt/beegfs/home/friesner/bgl2126/schrod_envs/laloosae/bin/activate

# Tells programs how many CPUs are available
export OMP_NUM_THREADS=$SLURM_NTASKS

# Bypass proxy for the Schrodinger license server
export no_proxy="$no_proxy,friesner.theo.chem.columbia.edu,10.198.22.10"
export NO_PROXY="$NO_PROXY,friesner.theo.chem.columbia.edu,10.198.22.10"

cd "$(dirname "$0")/.."
python scripts/train_sae.py "$@"
