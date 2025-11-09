#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -J jup_server
#SBATCH -o jup_server.out

# Load schrodinger and activate environment
module unload schrodinger; export SCHRODINGER=/cm/shared/apps/schrodinger/builds/NB/2025-4/build-011
source /mnt/beegfs/home/friesner/yw4484/schrod_py_envs/schrod3_build_011/bin/activate

# Tells programs how many CPUs are available
export OMP_NUM_THREADS=$SLURM_NTASKS

# This starts jupyter server
jupyter lab --no-browser --ip=$(hostname -s)








