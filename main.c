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
	int threadNum;

	SparseMatrix* A_local;
	double* x_local;
	double* x_global;
	double* b_local;
	double* r_local;
	double* y_local;
	double* diagInv_local;
	double tmpDoubleStore;
	double normB_global;
	double normR_global;

	double timer0 = 0;
	double timer1 = 0;
	double timer2 = 0;	

	/* *** START CODING *** */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	// Checks number of arguments
	if (argc <= 2)
	{
		if (myRank == 0)
		{
			fprintf(stderr,"Error: needs two arguments\n");
		}
		MPI_Finalize();
		return -1;
	}

	// Checks the correct type of input
	char* ptr;
	char* ptr2;
	dim = (int) strtol(argv[1],	&ptr, 10);
	threadNum = (int) strtol(argv[2], &ptr2, 10);
	if (*ptr != '\0')
	{
		if (myRank == 0)
		{
			fprintf(stderr,"Error: needs an integer on argument 1\n");
		}
		MPI_Finalize();
		return -1;
	}

	if (*ptr2 != '\0')
	{
		if (myRank == 0)
		{
			fprintf(stderr, "Error: needs an interger on argument 2\n");
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


	// Get x values for multiplication ** LOCAL **
	x_local = (double*) malloc (sizeof(double)*(numRow + 2));
	for (i = 0; i < numRow + 2; i++)
	{
		x_local[i] = 0.0;
	}

	// Get b values ** LOCAL **
	b_local = (double*) malloc (sizeof(double)*numRow);
	for (i = 0; i < numRow; i++)
	{
		b_local[i] = 1.0;
	}
	// DUMMY VALUES
	if (myRank == 0)
	{
		x_local[0] = -2000;
	}
	if (myRank == pNum -1)
	{
		x_local[numRow+1] = -2000;
	}

	// Get diagonal matrix ** LOCAL **
	diagInv_local = (double*) malloc (sizeof(double)*numRow);

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

	MPI_Allreduce(&tmpDoubleStore, &normB_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// Initialize y and r
	r_local = (double*) malloc (sizeof(double)*numRow);
	y_local = (double*) malloc (sizeof(double)*numRow);
	double* x_1_thread = (double*) malloc(sizeof(double)*numRow);

	// x_1_thread for 1 thread
	for (i = 0; i < numRow; i++)
	{
		x_1_thread[i] = 0.0;
	}


	int status;
	int myFirstI;
	int myLastI;

	int k;
	int k_new;
	int kmax;

	if (dim < 268000)
	{
		kmax = 8000 * dim;
	}
	else
	{
		/* Use the largest integral value - 1 */
		kmax = 214783646; 
	}  

	timer0 -= getTime();
	for (k = 1; k < kmax; k++)
	{ 
		k_new = k;
		// MULTIPLICATION PROCESS
		timer1 -= getTime();
		if (pNum > 1)
		{
			if (myRank == 0)
			{
				MPI_Send(&x_local[numRow], 1, MPI_DOUBLE, myRank+1, 99, MPI_COMM_WORLD);
				MPI_Recv(&x_local[numRow+1], 1, MPI_DOUBLE, myRank+1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
			}
			else if (myRank == pNum-1 && myRank != 0)
			{
				MPI_Send(&x_local[1], 1, MPI_DOUBLE, myRank-1, 99, MPI_COMM_WORLD);
				MPI_Recv(&x_local[0], 1, MPI_DOUBLE, myRank-1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			else
			{
				MPI_Send(&x_local[1], 1, MPI_DOUBLE, myRank-1, 99, MPI_COMM_WORLD);
				MPI_Send(&x_local[numRow], 1, MPI_DOUBLE, myRank+1, 99, MPI_COMM_WORLD);
				MPI_Recv(&x_local[numRow+1], 1, MPI_DOUBLE, myRank+1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&x_local[0],1, MPI_DOUBLE, myRank-1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
			}
		}
		double sum_local;
		int index;

		for (i = 0; i < numRow; i++)
		{
			sum_local = 0.0;
			for (j = A_local->iRow[i]; j < A_local->iRow[i+1]; j++)
			{
				if (pNum  > 1)
				{
					if (myRank == 0)
					{
						index = A_local->jCol[j] + 1;
					} 
					else
					{
						index = A_local->jCol[j] - (rowOrder - 1);
					}

					sum_local += A_local->values[j] * x_local[index];
				}
				else
				{
					sum_local += A_local->values[j] * x_1_thread[A_local->jCol[j]];
				}	
			}
			r_local[i] = sum_local;
		} 
		int k,l;


		// UPDATE RESIDUAL
		for (j = 0; j < numRow; j++)
		{
			r_local[j] = b_local[j] - r_local[j];
		}

		// CALCULATE R NORM
		normR_global = 0.0;
		for (j = 0; j < numRow; j++)
		{
			normR_global += r_local[j]*r_local[j];
		}
		timer1 += getTime();

		tmpDoubleStore = normR_global;

		MPI_Allreduce(&tmpDoubleStore, &normR_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		/*if (myRank == 0)
		  printf("Iteration: %d, Norm Reduction: %e\n",k_new-1, sqrt(normR_global/normB_global));*/ 

		// Break off point
		if (normR_global < 1.0e-8 * normB_global)
		{
			if (myRank == 0)
			{
				printf("Iteration: %d, Norm Reduction: %e\n",k_new-1, sqrt(normR_global/normB_global));
			}
			break;
		}


		// UPDATE X
		timer2 -= getTime();	
		if (pNum == 1)
		{
			for (j = 0; j < numRow; j++)
			{
				x_1_thread[j] += diagInv_local[j] * r_local[j];
			}
		}
		else
		{
			for (j = 0; j < numRow; j++)
			{
				x_local[j+1] += diagInv_local[j] * r_local[j];
			}

		}
		timer2 += getTime();


		// UPDATE X
		for (i = 0; i < numRow; i++)
		{
			y_local[i] = x_local[i+1];
		}

		// Outputs to file
		if (pNum < 2)
		{
			if (k_new==1)
			{
				out = fopen("x1.txt", "w");
				for (i = 0; i < dim; i++)
					fprintf(out, "%e\n", x_1_thread[i]);
				fclose(out);

			}
			if (k_new==2)
			{
				out = fopen("x2.txt", "w");
				for (i = 0; i < dim; i++)
					fprintf(out, "%e\n", x_1_thread[i]);
				fclose(out);

			}
		}
		else
		{
			// OUTPUT TO FILE
			if (k_new == 1 || k_new == 2)
			{ 
				if (myRank == 0)
				{
					x_global = (double*)malloc(sizeof(double)*dim);	

					MPI_Gather(y_local, numRow, MPI_DOUBLE, x_global, numRow, MPI_DOUBLE, 0,  MPI_COMM_WORLD);
					//printf("K = %i\n", k_new);
					if (k_new == 1)
					{
						out = fopen("x1.txt", "w");
						for (i = 0; i < dim; i++)
							fprintf(out, "%e\n", x_global[i]);
						fclose(out);
					}
					if (k_new == 2)
					{
						out = fopen("x2.txt", "w");
						for (i = 0; i < dim; i++)
							fprintf(out, "%e\n", x_global[i]);
						fclose(out);
					}
					free(x_global);
				}
				else
				{
					MPI_Gather(y_local, numRow, MPI_DOUBLE, x_global, numRow, MPI_DOUBLE, 0,  MPI_COMM_WORLD);

				}
			}
		}

	}
	timer0 += getTime();

	tmpDoubleStore = timer0;
	MPI_Reduce(&tmpDoubleStore, &timer0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	tmpDoubleStore = timer1;
	MPI_Reduce(&tmpDoubleStore, &timer1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	tmpDoubleStore = timer2;
	MPI_Reduce(&tmpDoubleStore, &timer2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	// Printing time
	if (myRank == 0)

	{
		printf("\n ... Time Total: %e s (%e, %e) \n\n",
				timer0, timer1, timer2);
		out = fopen("time.txt", "w");
		fprintf(out, "%e %e %e\n", timer0, timer1, timer2);
		fclose(out);
	}
	if (pNum < 2)
	{
		out = fopen("xfinal.txt", "w");
		for (i = 0; i < numRow; i++)
			fprintf(out, "%e\n", x_1_thread[i]);
		fclose(out);

	}
	else
	{ 
		if (myRank == 0)
		{
			x_global = (double*)malloc(sizeof(double)*dim);	

			MPI_Gather(y_local, numRow, MPI_DOUBLE, x_global, numRow, MPI_DOUBLE, 0,  MPI_COMM_WORLD);

			out = fopen("xfinal.txt", "w");
			for (i = 0; i < dim; i++)
				fprintf(out, "%e\n", x_global[i]);
			fclose(out);

			free(x_global);
		}
		else
		{
			MPI_Gather(y_local, numRow, MPI_DOUBLE, x_global, numRow, MPI_DOUBLE, 0,  MPI_COMM_WORLD);

		}
	}



	freeMatrix(A_local);
	free(x_1_thread);	
	free(x_local);
	free(b_local);
	free(diagInv_local);
	free(r_local);
	free(y_local);

	MPI_Finalize();


	return 0;
}
