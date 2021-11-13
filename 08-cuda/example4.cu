// =================================================================
//
// File: example4.cu
// Author(s):
// Description: This file contains the code to count the number of
//				even numbers within an array using CUDA.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 1000000000 //1e9
#define THREADS	256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))
// implement your code
__global__ void even(int *array, int *results) {

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	cout << "result = " << *results;
	while (tid < SIZE) {
		//aux = (aux < array[tid])? aux : array[tid];
    if((array[tid]%2)==0){
      *results ++;
    }
		tid += blockDim.x * gridDim.x;
	}

}

int main(int argc, char* argv[]) {
	int i, *a, *results;
  int *d_a, *d_r;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);

  //results = (int *) malloc(sizeof(int) * BLOCKS);
  results = 0;

	cudaMalloc( (void**) &d_a, SIZE * sizeof(int) );
	cudaMalloc( (void**) &d_r, BLOCKS * sizeof(int) );

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, results, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	printf("Starting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
		start_timer();

		even<<<BLOCKS, THREADS>>> (d_a, d_r);

		ms += stop_timer();
	}

	cudaMemcpy(results, d_r, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

	/*int aux = INT_MAX;
	for (i = 0; i < BLOCKS; i++) {
		aux = MMIN(aux, results[i]);
	}*/

	printf("result = %i\n", results);
	printf("avg time = %.5lf\n", (ms / N));

	cudaFree(d_r);
	cudaFree(d_a);

	free(a);
  free(results);
	return 0;
}
