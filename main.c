#include <time.h> // for clock_gettime()
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
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
// column major index mapping
#define colmajmap(n,i,j,k) ((n)*nx*ny*nz+(k)*nx*ny+(j)*nx+(i))

void Ibreak_point(int index)
{
    // return;
    int nproc;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    printf("rank = %2d, Ibreaking point %d\n",rank, index);
    // usleep(50000);
}


void cblas_zgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
        const MKL_INT m, const MKL_INT n, const MKL_INT k, const void *alpha, const void *a, const MKL_INT lda, 
        const void *b, const MKL_INT ldb, const void *beta, void *c, const MKL_INT ldc);


/**
 * @brief Generate a test vector.
 */
void set_up_vector(double *X, const int nx, const int ny, 
    const int nz, const int ncol)
{
    double min_val = -0.5, max_val = 2.0;
    for (int n = 0; n < ncol; n++)
    {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    // X[colmajmap(i,j,k)] = randnum(min_val,max_val);
                    X[colmajmap(n,i,j,k)] = sin(colmajmap(n,i,j,k));
                    // X[colmajmap(i,j,k)] = 1.0;
                }
            }
        }
    }
}



/**
 * @brief Generate a test vector.
 */
void set_up_vector_complex(double complex *X, const int nx, const int ny, 
    const int nz, const int ncol)
{
    double min_val = -0.5, max_val = 2.0;
    for (int n = 0; n < ncol; n++)
    {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    // X[colmajmap(i,j,k)] = randnum(min_val,max_val);
                    int ind = colmajmap(n,i,j,k);
                    X[ind] = sin(ind) + cos(ind) * I;
                }
            }
        }
    }
}


double diff_complex_arrays(
    const double complex *a, const double complex *b, const int len)
{
    double err = 0.0;
    for (int i = 0; i < len; i++) {
        err = max(err, cabs(a[i] - b[i]));
    }
    return err;
}


/**
 * @brief Print a complex matrix A (showing only the mxn principle submatrix).
 * 
 * @param A The matrix to be printed.
 * @param ldai The step size in i dimension of the matrix A.
 * @param ldaj The step size in i dimension of the matrix A.
 * @param m Row size of the submatrix to be shown.
 * @param n Column size of the submatrix to be shown.
 */
void print_complex_matrix(double complex *A, int ldai, int ldaj, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double complex a = A[i*ldai+j*ldaj];
            printf("%10.6f + %10.6fi  ", creal(a), cimag(a));
        }
        printf("\n");
    }
}

/**
 * @brief Print a real matrix A (showing only the mxn principle submatrix).
 * 
 * @param A The matrix to be printed.
 * @param ldai The step size in i dimension of the matrix A.
 * @param ldaj The step size in i dimension of the matrix A.
 * @param m Row size of the submatrix to be shown.
 * @param n Column size of the submatrix to be shown.
 */
void print_real_matrix(double *A, int ldai, int ldaj, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double a = A[i*ldai+j*ldaj];
            printf("%10.6f ", a);
        }
        printf("\n");
    }
}


/**
 * @brief Double-type matrix times a complex double-typed matrix.
 *
 *        Find C = A x B, where A is real, B is complex, and thus the result C is complex.
 *        We first find real(C) = A x real(B), then find imag(C) = A x imag(C), then we set
 *        C = real(C) + imag(C) * I.
 */
void d_z_gemm_orig(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const double *a, const MKL_INT lda, 
    const void *b, const MKL_INT ldb, const double beta, void *c, const MKL_INT ldc)
{
    double __complex__ *B = (double __complex__ *)b;
    double __complex__ *C = (double __complex__ *)c;

    size_t matsize = m * n;
    // A * real(B)
    double *b_real = malloc(k * n * sizeof(double));
    double *c_real = malloc(matsize * sizeof(double));
    assert(c_real != NULL && b_real != NULL);

    for (int i = 0; i < k * n; i++) b_real[i] = creal(B[i]);

    // find c_real = a * real(b)
    cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b_real, ldb, beta, c_real, ldc);

    // save real part of c into c
    for (int i = 0; i < matsize; i++) C[i] = c_real[i];

    double *b_imag = b_real;
    double *c_imag = c_real;
    for (int i = 0; i < k * n; i++) b_imag[i] = cimag(B[i]);

    // find c_imag = a * imag(b)
    cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b_imag, ldb, beta, c_imag, ldc);

    // save imag part of c into c
    for (int i = 0; i < matsize; i++) C[i] += c_imag[i] * I;

    free(b_real);
    free(c_real);
}



/**
 * @brief Double-type matrix times a complex double-typed matrix.
 *
 *        Find C = A x B, where A is real, B is complex, and thus the result C is complex.
 *        We first find real(C) = A x real(B), then find imag(C) = A x imag(C), then we set
 *        C = real(C) + imag(C) * I.
 */
void d_z_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const MKL_INT m, const MKL_INT n, const MKL_INT k, const void *alpha, const double *a, const MKL_INT lda, 
    const void *b, const MKL_INT ldb, const void *beta, void *c, const MKL_INT ldc)
{
    double __complex__ *B = (double __complex__ *)b;
    double __complex__ *C = (double __complex__ *)c;
    double __complex__ Alpha = *((double __complex__ *)alpha);
    double __complex__ Beta = *((double __complex__ *)beta);

    size_t matsize = m * n;
    // A * real(B)
    double *b_real = malloc(k * n * sizeof(double));
    double *c_real = malloc(matsize * sizeof(double));
    assert(c_real != NULL && b_real != NULL);

    for (int i = 0; i < k * n; i++) b_real[i] = creal(B[i]);

    // find c_real = a * real(b)
    cblas_dgemm(Layout, transa, transb, m, n, k, 1.0, a, lda, b_real, ldb, 0.0, c_real, ldc);

    // save real part of c into c
    if (cabs(Beta) < 1e-12) {
        for (int i = 0; i < matsize; i++) C[i] = Alpha * c_real[i];
    } else {
        for (int i = 0; i < matsize; i++) C[i] = Alpha * c_real[i] + Beta * C[i];
    }

    double *b_imag = b_real;
    double *c_imag = c_real;
    for (int i = 0; i < k * n; i++) b_imag[i] = cimag(B[i]);

    // find c_imag = a * imag(b)
    cblas_dgemm(Layout, transa, transb, m, n, k, 1.0, a, lda, b_imag, ldb, 0.0, c_imag, ldc);

    // save imag part of c into c
    double __complex__ Alpha_i = Alpha * I;
    for (int i = 0; i < matsize; i++) C[i] += c_imag[i] * Alpha_i;

    free(b_real);
    free(c_real);
}




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

    // ** Matrix size ** //
	int M, N, K;
	// Global matrix size
	// M = 1000; // number of rows of the global matrix
	// N = 1000; // number of columns of the global matrix
    // K = 2500;
    // M = 80, N = 80, K = 100;
    M = 8, N = 10, K = 2000;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    printf(YEL "M = %d, N = %d, K = %d\n" RESET, M, N, K);



	//A = (double *)malloc( sizeof(double) * M_loc * N_loc);
	double *A_real = (double *)calloc(M * K, sizeof(double));
	double complex *A = (double complex *)calloc(K * M, sizeof(double complex));
	double complex *B = (double complex *)calloc(K * N, sizeof(double complex));
	double complex *C = (double complex *)calloc(M * N, sizeof(double complex));
	double complex *C_ref = (double complex *)calloc(M * N, sizeof(double complex));
	assert(A != NULL && B != NULL);
    assert(C != NULL && C_ref != NULL);
// Ibreak_point(0);

    srand(1);
    set_up_vector(A_real, K, 1, 1, M);
    for (int i = 0; i < M*K; i++) A[i] = A_real[i];
    set_up_vector_complex(B, K, 1, 1, N);
// Ibreak_point(1);

    // printf("A = \n");
    // print_real_matrix(A_real, 1, K, 8, 8);
    // printf("B = \n");
    // print_complex_matrix(B, 1, K, 8, 8);
// Ibreak_point(2);


    // double complex alpha = 1.0, beta = 0.0;
    double complex alpha = 0.5+1.1*I, beta = 1.1+0.5*I;
    // warm-up
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, &alpha, A, M, B, K, &beta, C, M);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, &alpha, A, M, B, K, &beta, C_ref, M);

// Ibreak_point(3);

    // find A * B (A real, B complex), using two methods
    // method 1: cblas_zgemm, method 2: d_z_gemm
    // METHOD 1:
    const int nrep = 10;
    double t_tot;
    t1 = MPI_Wtime();
    for (n = 0; n < nrep; n++) {
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            M, N, K, &alpha, A, K, B, K, &beta, C_ref, M);
    }
    t2 = MPI_Wtime();
    t_tot = (t2 - t1)/nrep * 1000;
    printf(GRN "Time for zgemm: %.3f ms\n" RESET, t_tot);

    // printf("C_ref = \n");
    // print_complex_matrix(C_ref, 1, M, 8, 8);

    // METHOD 2:
    t1 = MPI_Wtime();
    for (n = 0; n < nrep; n++) {
        // d_z_gemm_orig(CblasColMajor, CblasTrans, CblasNoTrans,
        //     M, N, K, 1.0, A_real, K, B, K, 0.0, C, M);
        d_z_gemm(CblasColMajor, CblasTrans, CblasNoTrans,
            M, N, K, &alpha, A_real, K, B, K, &beta, C, M);
    }
    t2 = MPI_Wtime();
    t_tot = (t2 - t1)/nrep * 1000;
    printf(GRN "Time for d_z_gemm: %.3f ms\n" RESET, t_tot);

    // printf("C = \n");
    // print_complex_matrix(C, 1, M, 8, 8);

    double err = diff_complex_arrays(C_ref, C, M*N);
    printf("err = %.3e\n", err);

    // for (int i = 0; i < M*N; i++) {
    //     err = cabs(C_ref[i] - C[i]);
    //     if (err > 10) {
    //         printf("i = %d, C_ref = %f + %fi, C = %f + %fi\n",
    //             i, creal(C_ref[i]), cimag(C_ref[i]),
    //             creal(C[i]), cimag(C[i]));
    //     }
    // }


    free(A_real);
	free(A);
	free(B);
	free(C);
	free(C_ref);
	
	MPI_Finalize();
	return 0;	
}



