#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J preprocess_data
#SBATCH -o preprocess_data_%j.out

# Load schrodinger and activate environment
module unload schrodinger; export SCHRODINGER=/cm/shared/apps/schrodinger/builds/NB/2025-4/build-050
source /mnt/beegfs/home/friesner/bgl2126/schrod_envs/laloosae/bin/activate

# Tells programs how many CPUs are available
export OMP_NUM_THREADS=$SLURM_NTASKS

# Bypass proxy for the Schrodinger license server
export no_proxy="$no_proxy,friesner.theo.chem.columbia.edu,10.198.22.10"
export NO_PROXY="$NO_PROXY,friesner.theo.chem.columbia.edu,10.198.22.10"

cd "$SLURM_SUBMIT_DIR"
python scripts/preprocess_data.py "$@"
