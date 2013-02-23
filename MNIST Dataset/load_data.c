#include "mex.h"
#include "io64.h"

void timestwo(double y[], double x[])
{
  y[0] = 2.0*x[0];
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  
    size_t mrows, ncols;
    char buf[100];
    FILE * fp;
             
    int magic_number;
             
    #define DATA_OUT plhs[0]  
    #define TRAIN_DATA prhs[0]
    #define TRAIN_DATA_LABELS prhs[1]  
    #define TEST_DATA prhs[2]
    #define TEST_DATA_LABELS prhs[3]       
  
    if (nrhs > 0 && !mxIsChar(TRAIN_DATA)){
         mexErrMsgIdAndTxt( "MATLAB:load_data:inputNotChar",
            "First input must be a char vector.");
    }
    /* The input must be a noncomplex scalar double.*/
    
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    
    mxGetString(TRAIN_DATA, buf, ncols+1);
    mexPrintf(buf);
    mexPrintf("\n");
    fp = fopen(buf, "r+b");
    if (NULL == fp)
       mexPrintf("ERROR");
    else
    {
        
        int n = fread((void*)&magic_number, sizeof(int), 1, fp);
        mexPrintf("%d %d\n",n,magic_number); //should be 1 2051
        n = fread((void*)&magic_number, sizeof(int), 1, fp);
        mexPrintf("%d %d\n",n,magic_number); //should be 1 2051
        fclose(fp);
    }
    
          
          
//    /* Check for proper number of arguments. */
//   if(nrhs!=1) {
//    
//   } else if(nlhs>1) {
//     mexErrMsgIdAndTxt( "MATLAB:timestwo:maxlhs",
//             "Too many output arguments.");
//   }
//           
//     
//   double *x,*y;
// 
//   
//  
//   
// 
//   
//   //B OUT = mxCreateDoubleMatrix(M, N, mxREAL); /* Create the output matrix */
//   //B = mxGetPr(B OUT); /* Get the pointer to the data of B */
//   if( !mxIsChar(prhs[0]) )
//         mexErrMsgIdAndTxt( "MATLAB:timestwo:inputNotRealScalarDouble",
//             "Input must be a string");  
//   
//   if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
//       !(mrows==1 && ncols==1) ) {
//     mexErrMsgIdAndTxt( "MATLAB:timestwo:inputNotRealScalarDouble",
//             "Input must be a noncomplex scalar double.");
//   }
//   
//   /* Create matrix for the return argument. */
//   plhs[0] = mxCreateDoubleMatrix((mwSize)mrows, (mwSize)ncols, mxREAL);
//   
//   /* Assign pointers to each input and output. */
//   x = mxGetPr(prhs[0]);
//   y = mxGetPr(plhs[0]);
//   
//   /* Call the timestwo subroutine. */
//   timestwo(y,x);
}