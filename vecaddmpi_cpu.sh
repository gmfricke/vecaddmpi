#!/bin/bash

#SBATCH --ntasks=4

module load openmpi/4.1.2-q2zi

srun ./vecaddmpi
