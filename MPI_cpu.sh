#!/bin/bash
#SBATCH --job-name=jacobi_scaling
#SBATCH --partition=THIN
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24     # max MPI tasks per node
#SBATCH --time=02:00:00
#SBATCH --error=jacobi_%j.err
#SBATCH --exclusive

module load openMPI/4.1.6

# Problem size and iterations
N=1000
ITERS=1000
THREADS=1

# CSV output file
datafile="MPI_results.csv"
# Write header
echo "P,Initialization,Communication,Computation" > ${datafile}

# Loop over different numbers of MPI tasks
for P in 1 2 4 8 16 32 48; do
  echo "Running with ${P} MPI tasks..."
  # Capture timing output
  out=$(mpirun -np ${P} \
       ./jacobi_cpu ${N} ${ITERS} ${THREADS}
)
  # Extract times
  init_time=$(echo "${out}" | grep "Initialization time"    | awk '{print $3}')
  comm_time=$(echo "${out}" | grep "Communication time"   | awk '{print $3}')
  comp_time=$(echo "${out}" | grep "Computation time"     | awk '{print $3}')
  # Append to CSV
  echo "${P},${init_time},${comm_time},${comp_time}" >> ${datafile}
  echo "Results for P=${P}: init=${init_time}, comm=${comm_time}, comp=${comp_time}"
done

echo "Scaling runs complete. Results in ${datafile}"
