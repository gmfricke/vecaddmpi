#!/bin/bash
#SBATCH --job-name=vecaddmpi_gpu
#SBTACH --partition=dualGPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --gpus 2
#SBATCH --mail-user=yourusername@email.addr
#SBATCH --mail-type=All
#SBATCH --output=vecaddmpi_gpu.out

module load openmpi cuda
srun vecadd_mpi_gpu

