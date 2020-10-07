#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
 
/* Program Parameters */
#define MAXN 10000  /* Max value of N */
int size;  /* Matrix size */

 /* Input Matrix size */
void Input_Matrix(){
    printf("Input The Matrix Size (less than 10000):\n");
    scanf("%d",&size);
}

/* Matrices and vectors */
double A[MAXN][MAXN],U[MAXN][MAXN];
double B[MAXN],y[MAXN];
double X[MAXN];
/*  A*X = B, solve for X */

/* Initialize A and B, then, let X to 0.00 */
void initialize_inputs() {
  int i, j;
 
  printf("\nInitializing...\n");
  for (j = 0; j < size; j++) {
    for (i = 0; i < size; i++) {
        A[i][j] = (double)rand() / 1000.00;
    }
    B[j] = (double)rand() / 1000.00;
    X[j] = 0.00;
  }
}

/* Print input matrices */
void print_inputs() {
  int i, j;
 
  if (size < 30) {
    printf("\n Matrix A = \n");
    for (i = 0; i < size; i++) {
      for (j = 0; j < size; j++) {
        printf("%lf\t", A[i][j]);
      }
      printf("\n");
    }
    printf("\n Vector B = \n");
    for (j = 0; j < size; j++) {
      printf("%lf\t", B[j]);
    }
  }printf("\n");
}
 
void print_X() {
    int i;
 
    if (size < 30) {
        printf("\nX = \n");
        for (i = 0; i < size; i++) {
        printf("%lf", X[i]);
    }
    printf("]");
  }
}


void gaussian(){
    /* element row and col */
    int k, j, i;
    
    /* Correctly handle A[k,k]=0 */
    for (int i = 0; i < size; i++){
        if (A[i][i] == 0)
        {
            A[i][i] = 1;
        }
    }

    /* Gaussian elimination */
    for (k = 0; k < size; k++) {
        //Find the coefficients of the K-th elementary row transformation
        omp_set_num_threads(2);
        #pragma omp parallel for
        for (j = k + 1; j < size; j++) {
            A[k][j] = A[k][j] / A[k][k];}
        y[k]= B[k]/ A[k][k];
        A[k][k]= 1;
        omp_set_num_threads(2);
        #pragma omp parallel for
        //K-th elimination calculation
        for (j = k + 1; j < size; j++) {
            for (i = k+1; i < size; i++) {
                A[j][i] = A[j][i] - A[j][k] * A[k][i];
            }
            B[j] = B[j] - A[j][k] * y[k];///////////
            A[j][k] = 0;
        }
    }

    /* Back substitution */
    for (k = size - 1; k >= 0; k--) {
        X[k] = y[k];
        omp_set_num_threads(2);
        #pragma omp parallel for
        for (i = k-1; i > k; i--) {
            y[i] = y[i] - X[k] * A[i][k];
        }
    }
}
 
int main(int argc, char *argv[]) {
    /* Timing variables */
    struct timeval start, end;

    Input_Matrix();
    initialize_inputs();
    print_inputs();
    
      
    printf("\n Computing parallel.\n");
    /* Start Clock */
    gettimeofday(&start, NULL);
    
    gaussian();
    
    /* Display output */
    print_X();
    print_inputs();
    /* Display Time */
    gettimeofday(&end, NULL);/* End Clock */
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    printf("time spent=%12.10f\n",delta);

    exit(0);
}

