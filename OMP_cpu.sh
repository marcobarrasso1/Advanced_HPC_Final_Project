#!/bin/bash
#SBATCH --job-name=jacobi_scaling
#SBATCH --partition=EPYC
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --error=jacobi_%j.err
#SBATCH --exclusive

module load openMPI/4.1.6

# Problem size and iterations
N=1000
ITERS=1000
MPI_PROCESS=1

# CSV output file
datafile="OMP_results.csv"
# Write header
echo "Threads,Initialization,Communication,Computation" > ${datafile}

# Loop over different numbers of MPI tasks
for THREADS in 1 2 4 8 16 32 64 128 256 512; do
  echo "Running with ${P} MPI tasks..."
  # Capture timing output
  out=$(mpirun --map-by node --bind-to none -np ${MPI_PROCESS} \
       ./jacobi_cpu ${N} ${ITERS} ${THREADS}
)
  # Extract times
  init_time=$(echo "${out}" | grep "Initialization time"    | awk '{print $3}')
  comm_time=$(echo "${out}" | grep "Communication time"   | awk '{print $3}')
  comp_time=$(echo "${out}" | grep "Computation time"     | awk '{print $3}')
  # Append to CSV
  echo "${Threads},${init_time},${comm_time},${comp_time}" >> ${datafile}
done

echo "Scaling runs complete. Results in ${datafile}"
