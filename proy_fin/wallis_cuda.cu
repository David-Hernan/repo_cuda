/*----------------------------------------------------------------
*
* Multiprocesadores: Proyecto final
* Fecha: 24-Nov-2021
* Autor: A01173130 David Hernán García Fernández
*
*--------------------------------------------------------------*/

// ==============================================================
// Descripción: Este archivo contiene el código que, utilizando
//				CUDA, calcula una aproximación de PI mediante el
//				método del Producto de Wallis.
// ==============================================================

//Tiempo de ejecución secuencial:   ms
//Tiempo de ejecución con Threads:    ms
//Speed Up:

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 1000000000 //1e9
#define THREADS	256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))

__global__ void sum(int *array, double *result) {
	__shared__ long cache[THREADS];

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int cacheIndex = threadIdx.x;

	double acum = 1.0;
	while (tid < SIZE) {
		acum *= ((2.0*array[tid])/((2.0*array[tid])-1))*((2.0*array[tid])/((2.0*array[tid])+1));
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = acum;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i > 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		result[blockIdx.x] = cache[cacheIndex];
	}
}

int main(int argc, char* argv[]) {
	int i, *array, *d_a;
	double *results, *d_r;
	double ms;

	array = (int*) malloc( SIZE * sizeof(int) );
	for (i = 1; i < SIZE+1; i++) {
    array[i-1] = i;
  }

	results = (double*) malloc( BLOCKS * sizeof(double) );

	cudaMalloc( (void**) &d_a, SIZE * sizeof(int) );
	cudaMalloc( (void**) &d_r, BLOCKS * sizeof(double) );

	cudaMemcpy(d_a, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	printf("Starting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
		start_timer();

		sum<<<BLOCKS, THREADS>>> (d_a, d_r);

		ms += stop_timer();
	}

	cudaMemcpy(results, d_r, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

	double acum = 1.0;
	for (i = 0; i < BLOCKS; i++) {
		acum += results[i];
	}

	//printf("sum = %li\n", acum);
	printf("pi = %.16f para %d iteraciones.\n", acum,SIZE);
	printf("avg time = %.5lf\n", (ms / N));

	cudaFree(d_r);
	cudaFree(d_a);

	free(array);
	free(results);
	return 0;
}
