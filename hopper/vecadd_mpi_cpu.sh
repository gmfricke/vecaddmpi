#!/bin/bash
#SBATCH --job-name=vecaddmpi
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --mail-user=yourusername@email.addr
#SBATCH --mail-type=All

module load gcc/12.1.0-crtl
module load openmpi
srun vecadd_mpi_cpu

