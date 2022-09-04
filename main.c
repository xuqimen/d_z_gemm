#include <time.h> // for clock_gettime()
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "mpi.h"

#include "mkl.h"
#include <mkl_scalapack.h>
#include "mkl_lapacke.h"
#include <mkl_cblas.h>

#include <mkl_pblas.h>
//#include <mkl_scalapack.h>
#include <mkl_blacs.h>

// TO PRINT IN COLOR
#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))

int main(int argc, char **argv) {
	int i, j, k, n, p;
	double t1, t2, ts, te;	
	
	/************  MPI ***************************/
	int myrank_mpi, nprocs_mpi, rank, nproc;
	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
	/************  BLACS ***************************/
	// int ictxt_0, ictxt_1, ictxt_2, dims[2], nprow, npcol, myrow, mycol, nbr, nbc;
	int info,llda;
	int ZERO=0,ONE=1;
	
	// ** Process grid dimensions ** //
	// dims[0] = 40; dims[1] = 25;
	//dims[0] = 1; dims[1] = 1;
	
	// if (!rank && dims[0] * dims[1] != nproc) {
	//     printf("Error: dimensions of process grid is not consistent with the total number of processors!\n");
	//     exit(1);
	// }
	
	// ** Matrix size ** //
	int M, N;
	// Global matrix size
	M = 3500; // number of rows of the global matrix
	N = 3500; // number of columns of the global matrix
	// M = 8;
	// N = 8;

	double *A, *B, *v, *y, *y2;
	//A = (double *)malloc( sizeof(double) * M_loc * N_loc);
	A = (double *)mkl_malloc( sizeof(double) * max(1,M * N), 64);
	B = (double *)mkl_malloc( sizeof(double) * max(1,N * N), 64);
	v = (double *)mkl_malloc( sizeof(double) * max(1,N * 1), 64);
	y = (double *)mkl_malloc( sizeof(double) * max(1,N * 1), 64);
	y2 = (double *)mkl_malloc( sizeof(double) * max(1,N * 1), 64);
    
	assert(A != NULL && B != NULL && v != NULL && y != NULL && y2 != NULL);
	
    // set up random matrix and vector
  	srand(1+rank);
  	for (j = 0; j < N; j++) {
		for (i = 0; i < M; i++) {
			A[j*M+i] = 0.0 + (1.0 - 0.0) * (double) rand() / RAND_MAX;
		}
        v[j] = 0.0 + (1.0 - 0.0) * (double) rand() / RAND_MAX;
	}

    // set up symmetric matrix B = A' * A
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, N, M, 
                            1.0, A, M, A, M, 0.0, B, N);
	// ** Print out matrix ** //
    printf("\n-------------------------------------\n");
    printf(  "B:\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("%10.6f ", B[j*N+i]);
        }
        printf("\n");
    }
    printf("\n-------------------------------------\n");
    printf(  "v:\n");
    for (i = 0; i < 4; i++) {
        printf("%10.6f \n", v[i]);
    }

    // find A * x using two methods
    // method 1: cblas_dgemm, method 2: cblas_dsymv
    // METHOD 1:
    const int nrep = 6;
    double t_tot;
    t1 = MPI_Wtime();
    for (n = 0; n < nrep; n++) {
        v[0] = n;
        cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 
                1.0, B, N, v, 1, 0.0, y, 1);
    }
    t2 = MPI_Wtime();
    t_tot = (t2 - t1)/nrep * 1000;
    printf(GRN "Time for dgemv: %.3f ms\n" RESET, t_tot);
	// ** Print out matrix ** //
    printf("\n-------------------------------------\n");
    printf("GEMM y:\n");
    for (i = 0; i < 4; i++) {
        printf("%10.6f \n", y[i]);
    }

    
    // METHOD 2:
    t1 = MPI_Wtime();
    for (n = 0; n < nrep; n++) {
        cblas_dsymv(CblasColMajor, CblasUpper, N,
                1.0, B, N, v, 1, 0.0, y2, 1);
    }
    t2 = MPI_Wtime();
    t_tot = (t2 - t1)/nrep * 1000;
    printf(GRN "Time for dsymv: %.3f ms\n" RESET, t_tot);
	// ** Print out matrix ** //
    printf("\n-------------------------------------\n");
    printf("SYMV y:\n");
    for (i = 0; i < 4; i++) {
        printf("%10.6f \n", y2[i]);
    }


	mkl_free(A);
	mkl_free(B);
    mkl_free(v);
    mkl_free(y);
    mkl_free(y2);
	
	MPI_Finalize();
	return 0;	
}



