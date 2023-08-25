#!/bin/bash
#SBATCH --job-name=vecaddmpi_gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus 4
#SBATCH --mail-user=yourusername@email.addr
#SBATCH --mail-type=All
#SBATCH --output=vecaddmpi_gpu.out

module load openmpi cuda
srun vecadd_mpi_gpu

