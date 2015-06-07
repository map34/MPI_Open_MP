/*
 * =====================================================================================
 *
 *       Filename:  MyFunctions.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/05/2015 07:52:48 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "MyHeader.h"

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  CreateMatrix
 *  Description:  
 * =====================================================================================
 */
SparseMatrix* CreateMatrix(int myRank, int n, int pNum)
{
	int numRow = n / pNum;
	int rowOrder = myRank * numRow;
	// Allocating memory
	SparseMatrix* A = (SparseMatrix*)malloc(1*sizeof(SparseMatrix));
	A->m = numRow;
	A->n = n;
	A->iRow = (int*) malloc (sizeof(int)*(numRow+1));
	A->values = (double*) malloc (sizeof(double)*3*numRow);
	A->jCol = (int*) malloc (sizeof(int)*3*numRow);
	// Filling up array
	int i;
	A->iRow[0] = 0;
	for (i = 0; i < numRow; i++)
	{
		int row = i + rowOrder;
		if (row == 0)  
		{
			A->values[A->iRow[i]] = 2.0;
			A->jCol[A->iRow[i]] = row;
			A->values[A->iRow[i]+1] = -1.0;
			A->jCol[A->iRow[i]+1] = row+1;
			A->iRow[i+1] = A->iRow[i] + 2;
		}
		else if (row == n - 1)
		{
			A->values[A->iRow[i]] = -1.0;
			A->jCol[A->iRow[i]] = row-1;
			A->values[A->iRow[i]+1] = 2.0;
			A->jCol[A->iRow[i]+1] = row;
			A->iRow[i+1] = A->iRow[i] + 2;
		}
		else
		{
			A->values[A->iRow[i]] = -1.0;
			A->jCol[A->iRow[i]] = row - 1;
			A->values[A->iRow[i]+1] = 2.0;
			A->jCol[A->iRow[i]+1] = row;
			A->values[A->iRow[i]+2] = -1.0;
			A->jCol[A->iRow[i]+2] = row+1;
			A->iRow[i+1] = A->iRow[i] + 3;
		}
	}	
	return A;
}

/* -----  end of function CreateMatrix  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  freeMatrix
 *  Description:  
 * =====================================================================================
 */
void freeMatrix ( SparseMatrix* A  )
{
	free(A->values);
	free(A->iRow);
	free(A->jCol);
	free(A);
}		/* -----  end of function freeMatrix  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  JacobiMPI
 *  Description:  
 * =====================================================================================
 */
void JacobiMPI (SparseMatrix* A_local, double* x_global, double* b_local, int dim, int pNum, int myRank, MPI_Comm mpi_comm)
{

	FILE* out;
	int numRow = dim / pNum;
	int rowOrder = myRank * numRow;

	double normB_global;
	double normR_global;

	double tmpDoubleStore;

	double* y_local;
	double* r_local;

	double time0, time1, time2, time3, time4;


	// Get diagonal matrix ** LOCAL **
	double* diagInv_local = (double*) malloc (sizeof(double)*numRow);
	int i;
	for (i = 0; i < numRow; i++)
	{
		diagInv_local[i] = 0.5;
	}

	// Get norm of b ** LOCAL **
	for (i = 0; i < numRow; i++)
	{
		normB_global += b_local[i] *b_local[i];
	}
	tmpDoubleStore= normB_global;

	MPI_Allreduce(&tmpDoubleStore, &normB_global, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
	// Initialize y and r
	r_local = (double*) malloc (sizeof(double)*numRow);
	y_local = (double*) malloc (sizeof(double)*numRow);

	// JACOBI Process
	int j, l;
	double sum_local;
	time0 -= getTime();	// JACOBI TIMER
	for (i = 1; i < 2147483647; i++)
	{
		// y = A*x
		time1 -= getTime();		// MV Timer
		for (j = 0; j < numRow; j++)
		{
			sum_local = 0.0;
			for (l = A_local->iRow[j]; l < A_local->iRow[j+1]; l++)
			{
				sum_local += A_local->values[l] * x_global[A_local->jCol[l]];
			}
			y_local[j] = sum_local;
		}
		time1 += getTime();		// MV Timer

		// r = b - y, y = A*x 
		time2 -= getTime();		// R update timer
		for (j = 0; j < numRow; j++)
		{
			r_local[j] = b_local[j] - y_local[j];
		}
		time2 += getTime();		// R update timer

		// norm R calculations
		time3 -= getTime();		// Norm timer
		normR_global = 0.0;
		for (j = 0; j < numRow; j++)
		{
			normR_global += r_local[j]*r_local[j];
		}
		time3 += getTime();		// Norm timer
		tmpDoubleStore = normR_global;

		MPI_Allreduce(&tmpDoubleStore, &normR_global, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

		// Break off point
		if (normR_global < 1.0e-8 * normB_global)
		{
			if (myRank == 0)
			{
				printf("Iteration: %d, Norm Reduction: %e\n",i-1, sqrt(normR_global/normB_global));
			}
			break;
		}

		// y =  x + D^-1 * r
		time4 -= getTime();		// x update timer
		for (j = 0; j < numRow; j++)
		{
			y_local[j] = x_global[rowOrder + j] + diagInv_local[j] * r_local[j];
		}
		time4 += getTime();		// x update timer

		MPI_Allgather(y_local, numRow, MPI_DOUBLE, x_global, numRow, MPI_DOUBLE, mpi_comm);

		// output x1
		if (i == 1 && myRank == 0)
		{

			out = fopen("x1.txt", "w");
			for (j = 0; j < dim; j++)
			{
				fprintf(out,"%e\n",x_global[j]);
			}
			fclose(out);

		}

		// output x2
		if (i == 2 && myRank == 0)
		{

			out = fopen("x2.txt", "w");
			for (j = 0; j < dim; j++)
			{
				fprintf(out,"%e\n",x_global[j]);
			}
			fclose(out);

		}	
	}
	time0 += getTime();	// JACOBI TIME


	// REDUCE ALL TIMERS
	tmpDoubleStore = time0;
	MPI_Allreduce(&tmpDoubleStore, &time0, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

	tmpDoubleStore = time1;
	MPI_Allreduce(&tmpDoubleStore, &time1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

	tmpDoubleStore = time2;
	MPI_Allreduce(&tmpDoubleStore, &time2, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

	tmpDoubleStore = time3;
	MPI_Allreduce(&tmpDoubleStore, &time3, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

	tmpDoubleStore = time4;
	MPI_Allreduce(&tmpDoubleStore, &time4, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

	if (myRank == 0)
	{
		printf("---------Timer Results--------\n");
		printf("Jacobi: %e\nA*x: %e\nR update: %e\nNorm R: %e\nX update: %e\n", time0, time1, time2, time3, time4);
		out = fopen("time.txt", "w");
		fprintf(out, "%e %e %e %e %e\n", time0, time1, time2, time3, time4);
		fclose(out);
	}

	free(r_local);
	free(y_local);
	free(diagInv_local);


}		/* -----  end of function JacobiMPI  ----- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  getTime
 *  Description:  
 * =====================================================================================
 */

double getTime ( )
{


	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec + tp.tv_usec/1000000.0;


}		/* -----  end of function getTime  ----- */
