#!/bin/bash
#SBATCH --job-name=vecaddmpi
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=yourusername@email.addr
#SBATCH --mail-type=All
#SBATCH --output=vecaddmpi_cpu.out

module load openmpi
srun vecadd_mpi_cpu

