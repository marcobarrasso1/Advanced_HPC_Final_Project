#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <N> <iters>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int N     = atoi(argv[1]);  // # of interior rows & cols
    int iters = atoi(argv[2]);

    // printf("Hi form process: %d\n", rank);

    // 1) Domain decomposition
    int rows_per    = N / size;
    int extra_rows  = N % size;
    int local_rows  = rows_per + (rank < extra_rows ? 1 : 0);
    // start_row = global index of my first REAL row (0-based)
    int start_row = rank * rows_per + (rank < extra_rows ? rank : extra_rows);

    // 2) Allocate (local_rows+2) × (N+2) with ghost rows & cols
    int total_rows = local_rows + 2;
    int total_cols = N + 2;
    MPI_Aint bytes = sizeof(double) * total_rows * total_cols;

    double *mat     = NULL;
    double *mat_new = NULL;
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &mat);
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &mat_new);

    // 3) Zero everything
    memset(mat,     0, bytes);
    memset(mat_new, 0, bytes);

    // 4) Initialize interior to 0.5
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= N; j++) {
            mat[i*total_cols + j] = 0.5;
        }
    }

    // 5) Set Dirichlet BCs on global left & bottom, zero elsewhere
    double incr = 100.0/(N+1);

    // a) Left column j=0 for each real row
    for (int i = 1; i <= local_rows; i++) {
        double val = (start_row + i) * incr;
        mat    [i*total_cols + 0] = val;
        mat_new[i*total_cols + 0] = val;
    }

    // b) Bottom row only on last rank (padded row index = local_rows+1)
    if (rank == size - 1) {
        int brow = local_rows + 1;
        for (int j = 1; j <= N+1; j++) {
            double val = j * incr;
            int col = (N + 1) - j;
            mat    [brow*total_cols + col] = val;
            mat_new[brow*total_cols + col] = val;
        }
    }
    // Top ghost row (i=0) and right ghost col (j=N+1) remain zero from the memset

    // 6) Write out the initial matrix to "initial.bin"
    {
      MPI_File fh_init;
      MPI_File_open(MPI_COMM_WORLD,
                    "initial.bin",
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL,
                    &fh_init);

      // Compute byte‐offset so each rank writes into its correct location
      // We want to include the top‐ghost row exactly once (by rank 0)
      MPI_Offset file_offset;
      if (rank == 0) {
        // rank 0 writes its top‐ghost + all its real rows
        file_offset = 0;
        MPI_File_write_at_all(fh_init,
                              file_offset,
                              mat,
                              (local_rows + 1) * total_cols,
                              MPI_DOUBLE,
                              MPI_STATUS_IGNORE);
      } else {
        // all other ranks write only their real rows
        // padded row index to start = start_row + 1
        file_offset = (MPI_Offset)(start_row + 1) * total_cols * sizeof(double);

        if (rank == size - 1) {
          // last rank also writes its bottom ghost row
          MPI_File_write_at_all(fh_init,
                                file_offset,
                                &mat[1 * total_cols],
                                (local_rows + 1) * total_cols,
                                MPI_DOUBLE,
                                MPI_STATUS_IGNORE);
        } else {
          // interior ranks write just REAL rows
          MPI_File_write_at_all(fh_init,
                                file_offset,
                                &mat[1 * total_cols],
                                local_rows * total_cols,
                                MPI_DOUBLE,
                                MPI_STATUS_IGNORE);
        }
      }
      MPI_File_close(&fh_init);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 7) Set up two RMA windows for the first and last real rows
    MPI_Win win_top, win_bot;
    MPI_Win_create(&mat[1*total_cols],
                   total_cols*sizeof(double),
                   sizeof(double),
                   MPI_INFO_NULL,
                   MPI_COMM_WORLD,
                   &win_top);
    MPI_Win_create(&mat[local_rows*total_cols],
                   total_cols*sizeof(double),
                   sizeof(double),
                   MPI_INFO_NULL,
                   MPI_COMM_WORLD,
                   &win_bot);

    int above = (rank > 0 ? rank - 1 : MPI_PROC_NULL);
    int below = (rank < size-1 ? rank + 1 : MPI_PROC_NULL);

    // 8) Jacobi iterations
    for (int it = 0; it < iters; ++it) {
        // open epoch
        MPI_Win_fence(0, win_top);
        MPI_Win_fence(0, win_bot);

        // A) fetch below‐neighbor’s row 1 into my bottom ghost
        MPI_Get(&mat[(local_rows+1)*total_cols],
                total_cols, MPI_DOUBLE,
                below,  0,          total_cols, MPI_DOUBLE,
                win_top);

        // B) fetch above‐neighbor’s last real row into my top ghost
        MPI_Get(&mat[0*total_cols],
                total_cols, MPI_DOUBLE,
                above,  0,          total_cols, MPI_DOUBLE,
                win_bot);

        // close epoch
        MPI_Win_fence(0, win_top);
        MPI_Win_fence(0, win_bot);

        // local Jacobi sweep on interior only
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= N; j++) {
                mat_new[i*total_cols + j] = 0.25 * (
                    mat[(i-1)*total_cols + j] +
                    mat[(i+1)*total_cols + j] +
                    mat[i*total_cols + (j-1)] +
                    mat[i*total_cols + (j+1)]
                );
            }
        }
        // swap pointers
        double *tmp = mat; mat = mat_new; mat_new = tmp;
    }

    // 9) Write out the final matrix to "solution.bin"
    {
      MPI_File fh_sol;
      MPI_File_open(MPI_COMM_WORLD,
                    "solution.bin",
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL,
                    &fh_sol);

      MPI_Offset file_offset;
      if (rank == 0) {
        file_offset = 0;
        MPI_File_write_at_all(fh_sol,
                              file_offset,
                              mat,
                              (local_rows + 1) * total_cols,
                              MPI_DOUBLE,
                              MPI_STATUS_IGNORE);
      } else {
        file_offset = (MPI_Offset)(start_row + 1) * total_cols * sizeof(double);
        if (rank == size - 1) {
          MPI_File_write_at_all(fh_sol,
                                file_offset,
                                &mat[1 * total_cols],
                                (local_rows + 1) * total_cols,
                                MPI_DOUBLE,
                                MPI_STATUS_IGNORE);
        } else {
          MPI_File_write_at_all(fh_sol,
                                file_offset,
                                &mat[1 * total_cols],
                                local_rows * total_cols,
                                MPI_DOUBLE,
                                MPI_STATUS_IGNORE);
        }
      }
      MPI_File_close(&fh_sol);
    }

    // 10) Cleanup
    MPI_Win_free(&win_top);
    MPI_Win_free(&win_bot);
    MPI_Free_mem(mat);
    MPI_Free_mem(mat_new);
    MPI_Finalize();
    return 0;
}
