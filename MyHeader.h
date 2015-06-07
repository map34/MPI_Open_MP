/*
 * =====================================================================================
 *
 *       Filename:  MyHeader.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/05/2015 07:57:27 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

typedef struct {
  int m, n;
  int *iRow;
  int *jCol;
  double *values;
} SparseMatrix;

/* --- Create a sparse matrix --- */
SparseMatrix* CreateMatrix(int my_rank, int n, int comm_sz);


/* --- Timer Function --- */
double getTime();

/* --- Free Matrix --- */
void freeMatrix(SparseMatrix* A);

/* --- Jacobi MPI Process */
void JacobiMPI (SparseMatrix* A_local, double* x_global, double* b_local, int dim, int pNum, int myRank, MPI_Comm mpi_comm);
