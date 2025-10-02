#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
    // 1) Initialize MPI with OpenMP support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 || argc > 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <N> <iters> [num_threads]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N        = atoi(argv[1]);               // interior grid size
    int iters    = atoi(argv[2]);               // number of Jacobi iterations
    int nthreads = (argc == 4 ? atoi(argv[3]) : omp_get_max_threads());
    omp_set_num_threads(nthreads);

    if (rank == 0) {
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", size, nthreads);
    }

    // 2) Domain decomposition
    int rows_per   = N / size;
    int extra_rows = N % size;
    int local_rows = rows_per + (rank < extra_rows ? 1 : 0);
    int start_row  = rank * rows_per + (rank < extra_rows ? rank : extra_rows);

    // 3) Dimensions & allocation
    int total_rows = local_rows + 2;
    int total_cols = N + 2;
    int total_size = total_rows * total_cols;
    MPI_Aint bytes = sizeof(double) * total_size;

    double *mat     = NULL;
    double *mat_new = NULL;
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &mat);
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &mat_new);

    // 4) Host initialization timing
    MPI_Barrier(MPI_COMM_WORLD);
    double init_time = MPI_Wtime();

    memset(mat,     0, bytes);
    memset(mat_new, 0, bytes);

    // interior init
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 1; j <= N; ++j) {
            mat[i*total_cols + j] = 0.5;
        }
    }
    // boundary init
    double incr = 100.0 / (N + 1);
    #pragma omp parallel for
    for (int i = 1; i <= local_rows; ++i) {
        double v = (start_row + i) * incr;
        mat    [i*total_cols + 0] = v;
        mat_new[i*total_cols + 0] = v;
    }
    if (rank == size - 1) {
        int brow = local_rows + 1;
        #pragma omp parallel for
        for (int j = 1; j <= N+1; ++j) {
            double v = j * incr;
            int col = (N + 1) - j;
            mat    [brow*total_cols + col] = v;
            mat_new[brow*total_cols + col] = v;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    init_time = MPI_Wtime() - init_time;

    // 5) Setup neighbor ranks
    int above = (rank > 0 ? rank - 1 : MPI_PROC_NULL);
    int below = (rank < size-1 ? rank + 1 : MPI_PROC_NULL);

    // 6) Offload region with MPI RMA and GPU compute
    double communication_time = 0.0;
    double compute_time       = 0.0;

    #pragma omp target data map(tofrom: mat[0:total_size]) \
                              map(tofrom: mat_new[0:total_size])
    {
        // Create RMA windows on device pointer
        #pragma omp target data use_device_ptr(mat)
        {
            MPI_Win win_top, win_bot;
            MPI_Win_create(mat + (1*total_cols),          total_cols*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_top);
            MPI_Win_create(mat + (local_rows*total_cols), total_cols*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_bot);
        }
            for (int it = 0; it < iters; ++it) {
                // a) Exchange ghost rows via RMA

                #pragma omp barrier
                MPI_Barrier(MPI_COMM_WORLD);

                double t_comm = MPI_Wtime();

                #pragma omp target data use_device_ptr(mat)
                {
                    MPI_Win_fence(0, win_top);
                    MPI_Win_fence(0, win_bot);

                    MPI_Get(mat + ((local_rows+1)*total_cols), total_cols, MPI_DOUBLE,
                            below, 0, total_cols, MPI_DOUBLE, win_top);
                    MPI_Get(mat, total_cols, MPI_DOUBLE,
                            above, 0, total_cols, MPI_DOUBLE, win_bot);

                    MPI_Win_fence(0, win_top);
                    MPI_Win_fence(0, win_bot);
                }

                #pragma omp barrier
                MPI_Barrier(MPI_COMM_WORLD);

                communication_time += MPI_Wtime() - t_comm;

                // b) GPU compute
                double t_comp = MPI_Wtime();
                #pragma omp target teams distribute parallel for simd collapse(2) num_teams(108)
                for (int i = 1; i <= local_rows; ++i) {
                    for (int j = 1; j <= N; ++j) {
                        mat_new[i*total_cols + j] = 0.25 * (
                            mat[(i-1)*total_cols + j] +
                            mat[(i+1)*total_cols + j] +
                            mat[i*total_cols + (j-1)] +
                            mat[i*total_cols + (j+1)]
                        );
                    }
                }

                #pragma omp barrier
                MPI_Barrier(MPI_COMM_WORLD);

                compute_time += MPI_Wtime() - t_comp;

                // swap on device
                double *tmp = mat; mat = mat_new; mat_new = tmp;
            }

            // free windows
            MPI_Win_free(&win_top);
            MPI_Win_free(&win_bot);
        }
    }

    // 7) Print timings on rank 0
    if (rank == 0) {
        printf("Initialization time: %f s\n", init_time);
        printf("Communication time: %f s\n", communication_time);
        printf("Computation time (GPU): %f s\n", compute_time);
    }

    // 8) Cleanup
    MPI_Free_mem(mat);
    MPI_Free_mem(mat_new);
    MPI_Finalize();
    return 0;
}