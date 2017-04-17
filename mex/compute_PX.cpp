/* compute_PX.cpp Compute only sparse entries of matrix product
Syntax: PX = compute_PX(Q,R,rows,cols)
*/

#include <math.h>
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Macros for the ouput and input arguments */
	#define PX_OUT plhs[0]
	#define Q_IN prhs[0]
	#define R_IN prhs[1]
	#define ROWS_IN prhs[2]
	#define COLS_IN prhs[3]
	
	double *Q, *R, *PX, *ROWS, *COLS, inner_prod;
	int M, N, num_rows, numerical_rank;
	if(nrhs !=  4) {/* Check the number of arguments */
		mexErrMsgTxt("Incorrect number of input arguments.");
	}
	else if(nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}
	// Get dimensions
	M = mxGetM(Q_IN); /* Get the number of rows of Q */	
	N = mxGetN(R_IN); /* Get the number of cols of N */
	num_rows = mxGetM(ROWS_IN);
	numerical_rank = mxGetN(Q_IN); /* the numerical rank of our QR factorization */
	
	// Get pointers
	Q = mxGetPr(Q_IN); /* Get the pointer to the data of Q */
	R = mxGetPr(R_IN);
	ROWS = mxGetPr(ROWS_IN);
	COLS = mxGetPr(COLS_IN);

	PX_OUT = mxCreateDoubleMatrix(num_rows, 1, mxREAL); /* Create the output matrix */
	PX = mxGetPr(PX_OUT); /* Get the pointer to the data of PX */
	for(int i = 0; i < num_rows; i++) {
		inner_prod = 0;
		for(int j = 0; j < numerical_rank; j++) {
			inner_prod += Q[(int)ROWS[i] -1 + j*M]*R[j + numerical_rank*((int)COLS[i]-1)];
		}
		PX[i] = inner_prod;
	}    
	return;
}

