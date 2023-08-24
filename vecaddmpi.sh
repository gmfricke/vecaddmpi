#!/bin/bash
#SBATCH --job-name=vecaddmpi
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=mfricke@unm.edu
#SBATCH --mail-type=All
#SBATCH --output=vecaddmpi.out

module load openmpi/4.1.2-q2zi
srun ./vecaddmpi

