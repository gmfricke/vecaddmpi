#!/bin/bash
#SBATCH --job-name=vecaddmpi
#SBATCH --partition=singleGPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --mail-user=yourusername@email.addr
#SBATCH --mail-type=All

module load openmpi
srun vecadd_mpi_cpu

