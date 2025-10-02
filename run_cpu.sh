#!/bin/bash

#SBATCH --job-name=jacobi
#SBATCH --partition=THIN
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --error=jacobi%j.err
#SBATCH --exclusive

module load openMPI/4.1.6

export OMP_PLACES=cores 
export OMP_PROC_BIND=true

mpirun -np 2 ./jacobi_cpu 100 5000 8