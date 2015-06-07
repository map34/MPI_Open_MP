/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Jacobi Iteration using MPI
 *
 *        Version:  1.0
 *        Created:  05/31/2015 20:54:58
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Mochamad Prananda (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "MyHeader.h"
int main(int argc, char** argv)	
{
	int pNum;
	int myRank;
	int dim;
	int numRow;
	int rowOrder;
	FILE* out;

	SparseMatrix* A_local;
	double* x_global;
	double* b_local; 

	/* *** START CODING *** */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	// Checks number of arguments
	if (argc <= 1)
	{
		if (myRank == 0)
		{
			fprintf(stderr,"Error: needs an argument\n");
		}
		MPI_Finalize();
		return -1;
	}

	// Checks the correct type of input
	char* ptr;
	dim = (int) strtol(argv[1],	&ptr, 10);
	if (*ptr != '\0')
	{
		if (myRank == 0)
		{
			fprintf(stderr,"Error: needs an integer\n");
		}
		MPI_Finalize();
		return -1;
	}

	// Row numbers, and row order	
	numRow = dim / pNum;
	rowOrder = myRank * numRow;
	// get matrix
	int i,j;
	A_local = CreateMatrix(myRank, dim, pNum);

	// Get x values ** GLOBAL **
	x_global = (double*) malloc (sizeof(double)*dim);
	for (i = 0; i < dim; i++)
	{
		x_global[i] = 0.0;
	}

	// Get b values ** LOCAL **
	b_local = (double*) malloc (sizeof(double)*numRow);
	for (i = 0; i < numRow; i++)
	{
		b_local[i] = 1.0;
	}
	
	// Jacobi Process	
	JacobiMPI(A_local, x_global, b_local, dim, pNum, myRank, MPI_COMM_WORLD);

	// output xfinal
	if (myRank == 0)
	{
		out = fopen("xfinal.txt", "w");
		for (j = 0; j < dim; j++)
		{
			fprintf(out,"%e\n",x_global[j]);
		}
		fclose(out);
	}


	freeMatrix(A_local);
	free(x_global);
	free(b_local);

	MPI_Finalize();


	return 0;
}
