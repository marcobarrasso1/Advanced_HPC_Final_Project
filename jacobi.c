// Libraries
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char* argv[]){

  // Initialize MPI
  int mpi_provided_thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided_thread_level);
  if (mpi_provided_thread_level < MPI_THREAD_FUNNELED)
  {
    printf("A problem arised when asking for MPI_THREAD_FUNNELED level\n");
    MPI_Finalize();
    exit(1);
  }

  // Get the rank of the process and the number of processes
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get the number of OpenMP threads
  #pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    #pragma omp master
    {
      printf("Number of OpenMP threads: %d\n", num_threads);
    }
  }

  // Get the number of devices
  printf("Number of devices: %d\n", omp_get_num_devices());

  const int ngpus = omp_get_num_devices();
  const int device_id = rank % ngpus;
  omp_set_default_device(device_id);
  
  // Time measurements' variables
  double init_time = 0.0;
  double compute_time = 0.0;
  double communication_time = 0.0;
  double copy_time = 0.0;

  // Dimensions and iterations parameters
  size_t dimension = 0;
  size_t iterations = 0;

  // Check on input parameters
  if(argc != 3) {
    if (rank == 0) {
      fprintf(stderr, "\nWrong number of arguments. Usage: ./a.out dim it n\n");
    }
    MPI_Finalize();
    return 1;
  }

  // Read in the parameters
  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);

  // Check if the parameters are valid
  if (dimension <= 0 || iterations <= 0) {
    if (rank == 0) {
      fprintf(stderr, "\nInvalid parameters. Parameters must be positive integers.\n");
    }
    MPI_Finalize();
    return 1;
  }

  // Print the parameters
  if (rank == 0) {
    printf("Matrix size = %zu\n", dimension);
    printf("Number of iterations = %zu\n", iterations);
  }

  // Get the local amount of rows to be computed by each process
  // (Each process will compute a submatrix of the original matrix; the rows are distributed evenly among the processes:
  // the first processes will get one more row than the others if the number of rows is not divisible by the number of processes)
  const size_t rows_per_process = dimension / size;
  const size_t extra_rows = dimension % size;
  const size_t local_rows = (rank < (int)extra_rows) ? rows_per_process + 1 : rows_per_process;

  // Compute the offset (needed to access the correct rows in the global matrix)
  // The offset is the number of extra rows (if any) that the previous processes have taken
  const int offset = extra_rows * (rank >= (int)extra_rows);

  // Total number of rows for each process (including the ghost rows)
  const size_t total_rows = local_rows + 2;

  // Allocate memory for the matrices
  const MPI_Aint bytes = sizeof(double) * (total_rows) * (dimension + 2);
  double *matrix;
  double *matrix_new;
  MPI_Alloc_mem(bytes, MPI_INFO_NULL, &matrix);
  MPI_Alloc_mem(bytes, MPI_INFO_NULL, &matrix_new);

  // Sincronize the processes
  MPI_Barrier(MPI_COMM_WORLD);

  // Start the timer for initialization
  init_time = MPI_Wtime();

  // Initialize the matrices
  memset(matrix, 0, byte_dimension);
  memset(matrix_new, 0, byte_dimension);

  // Fill the initial values of the matrices with 0.5
  // (Initialization is done in parallel to speed up the process using OpenMP)
  #pragma omp parallel for collapse(2)
  for (size_t i = 1; i <= local_rows; ++i)
  {
    for (size_t j = 1; j <= dimension; ++j)
    {
      matrix[(i * (dimension + 2)) + j] = 0.5;
    }
  }
	      
  // Set up the boundary conditions
  // (Boundary conditions are set for the first column and the last row of the global matrix)
  // (Initialization is done in parallel to speed up the process using OpenMP)
  const double increment = 100.0 / ( dimension + 1 );

  // First row of each process
  int start_row = local_rows * rank + offset;

  // Last process will have to initialize the last row of the global matrix
  // All processes contribute to the initialization of the first column of the global matrix
  if (rank == size - 1) // Last process
  {
    // Vertical border
    #pragma omp parallel for
    for (size_t i = 1; i <= local_rows + 1; ++i)
    {
      matrix[i * (dimension + 2)] = (start_row + i) * increment;
      matrix_new[i * (dimension + 2)] = (start_row + i) * increment;
    }

    // Horizontal border
    #pragma omp parallel for
    for (size_t j = 1; j <= dimension + 1; ++j)
    {
      matrix[((local_rows + 1) * (dimension + 2)) + (dimension + 1 - j)] = j * increment;
      matrix_new[((local_rows + 1) * (dimension + 2)) + (dimension + 1 - j)] = j * increment;
    }
  }
  else // All other processes
  {
    // Vertical border
    #pragma omp parallel for
    for (size_t i = 1; i <= local_rows; ++i)
    {
      matrix[i * (dimension + 2)] = (start_row + i) * increment;
      matrix_new[i * (dimension + 2)] = (start_row + i) * increment;
    }
  }


  // Sincronize the processes
  MPI_Barrier(MPI_COMM_WORLD);

  // Stop the timer for initialization
  init_time = MPI_Wtime() - init_time;

  // Print the initialization time
  if (rank == 0) {
    printf("Initialization time: %f seconds\n", init_time);
  }

  // Define the "neighbors" for each process:
  // each process will need to "communicate" with its neighbors (the previous and next processes) to exchange the rows
  // needed for the computation of the new values. Remote memory access (RMA) will be used for this purpose.
  // The first and last processes will have MPI_PROC_NULL as their neighbor above and below respectively
  int proc_above = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  int proc_below = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

  // Create the two memory windows (remote memory access):
  // one for the first row and one for the last row of the local matrix
  // (Processes will use these windows to access the rows of their neighbors)
  MPI_Win win_top;
  MPI_Win win_bottom;

  const int total_size = total_rows * (dimension + 2); // Total size of the matrix to be used in the OpenMP target region
  
  // Sincronize the processes
  MPI_Barrier(MPI_COMM_WORLD);

  copy_time = MPI_Wtime(); // Start the timer for the copy time

  #pragma omp target data map(tofrom: matrix[0:total_size]) map(to: matrix_new[0:total_size])
  {

    // Syncronize openmp threads
    #pragma omp barrier

    // Synchronize the processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Stop the timer for the copy time
    copy_time = MPI_Wtime() - copy_time;

    if (rank == 0) {
      printf("Copy time: %f seconds\n", copy_time);
    }

    // Use device pointer for MPI_Win_create
    #pragma omp target data use_device_ptr(matrix)
    {
      // Create the memory windows for the first and last rows of the local matrix
      MPI_Win_create(&matrix[dimension + 2], (dimension + 2) * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_top);
      MPI_Win_create(&matrix[local_rows * (dimension + 2)], (dimension + 2) * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_bottom);
    }

    // Start the algorithm
    for (size_t it = 0; it < iterations; ++it)
    {

      // Sincronize the openmp threads
      #pragma omp barrier

      // Sincronize the processes
      MPI_Barrier(MPI_COMM_WORLD);

      // Start the timer for the communication time
      double t0 = MPI_Wtime();

      // Use the windows to access the rows of the neighbors
      #pragma omp target data use_device_ptr(matrix)
      {
        MPI_Win_fence(0, win_top); // Synchronize the top window
        MPI_Win_fence(0, win_bottom); // Synchronize the bottom window

        MPI_Get(&matrix[(local_rows + 1) * (dimension + 2)], (MPI_Aint) (dimension + 2), MPI_DOUBLE, proc_below,
                0, (MPI_Aint) (dimension + 2), MPI_DOUBLE, win_top); // Get the first row of the next process
        MPI_Get(matrix, (MPI_Aint) (dimension + 2), MPI_DOUBLE, proc_above,
                0, (MPI_Aint) (dimension + 2), MPI_DOUBLE, win_bottom); // Get the last row of the previous process

        MPI_Win_fence(0, win_top); // Synchronize the top window
        MPI_Win_fence(0, win_bottom); // Synchronize the bottom window
      }

      // Sincronize the openmp threads
      #pragma omp barrier

      // Sincronize the processes
      MPI_Barrier(MPI_COMM_WORLD);

      // Stop the timer for the communication time
      communication_time += MPI_Wtime() - t0;

      // Start the timer for the computation time
      double t1 = MPI_Wtime();

      // Compute the new values of the matrix
      #pragma omp target teams distribute parallel for simd collapse(2) num_teams(108)  // Use OpenMP to parallelize the computation
      for (size_t i = 1; i <= local_rows; ++i)
      {
        for (size_t j = 1; j <= dimension; ++j)
        {
          matrix_new[(i * (dimension + 2)) + j] = 0.25 * (matrix[((i - 1) * (dimension + 2)) + j] +
                                                          matrix[(i * (dimension + 2)) + (j + 1)] +
                                                          matrix[((i + 1) * (dimension + 2)) + j] +
                                                          matrix[(i * (dimension + 2)) + (j - 1)]);
        }
      }

      // Swap the matrices (matrix and matrix_new)
      #pragma omp target teams distribute parallel for simd collapse(2) num_teams(108)  // Use OpenMP to parallelize the computation
      for (size_t i = 1; i <= local_rows; ++i)
      {
        for (size_t j = 1; j <= dimension; ++j)
        {
          matrix[(i * (dimension + 2)) + j] = matrix_new[(i * (dimension + 2)) + j];
        }
      }

      // Sincronize the openmp threads
      #pragma omp barrier

      // Sincronize the processes
      MPI_Barrier(MPI_COMM_WORLD);

      // Stop the timer for the computation time
      compute_time += MPI_Wtime() - t1;

    } // End of iterations

  }

  // Free the memory windows
  MPI_Win_free(&win_top);
  MPI_Win_free(&win_bottom);

  // Print the communication, computation and total times
  if (rank == 0) {
    printf("Communication time: %f seconds\n", communication_time);
    printf("Computation time: %f seconds\n", compute_time);
    double total_time = init_time + compute_time + communication_time + copy_time;
    printf("Total time: %f seconds\n", total_time);
  }

  // Save the results to a file
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, "solution.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  // Handle the distribution of the data across the processes
  if (rank == 0) { // First process
    MPI_Offset fh_offset = 0;
    MPI_File_write_at_all(fh, fh_offset, matrix, (local_rows + 1) * (dimension + 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
  }
  else if (rank == size - 1) { // Last process
    MPI_Offset fh_offset = (rank * local_rows * (dimension + 2)) + (dimension + 2);
    // Handle the extra_rows
    if (rank >= (int)extra_rows) {
      fh_offset += extra_rows;
    }
    fh_offset *= sizeof(double);
    MPI_File_write_at_all(fh, fh_offset, &matrix[dimension + 2], (local_rows + 1) * (dimension + 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
  }
  else { // All other processes
    MPI_Offset fh_offset = (rank * local_rows * (dimension + 2)) + (dimension + 2);
    // Handle the extra_rows
    if (rank >= (int)extra_rows) {
      fh_offset += extra_rows;
    }
    fh_offset *= sizeof(double);
    MPI_File_write_at_all(fh, fh_offset, &matrix[dimension + 2], local_rows * (dimension + 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
  }
  MPI_File_close(&fh);
  
  // Free the allocated memory
  MPI_Free_mem(matrix);
  MPI_Free_mem(matrix_new);

  // Save the times to csv a file
  if (rank == 0) {
    FILE *fp = fopen("times.csv", "a+");
    if (fp == NULL) {
      fprintf(stderr, "Error opening file\n");
      MPI_Finalize();
      return 1;
    }
    fprintf(fp, "%zu,%zu,%f,%f,%f,%f\n", dimension, iterations, init_time, compute_time, communication_time, copy_time);
    fclose(fp);
  }

  // Finalize MPI
  MPI_Finalize();

  return 0;
}