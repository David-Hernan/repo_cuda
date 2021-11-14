/*----------------------------------------------------------------
*
* Multiprocesadores: CUDA
* Fecha: 14-Nov-2021
* Autores:
* A01173130 David Hernán García Fernández
* A01701434 Joseph Alessandro García García
*
*--------------------------------------------------------------*/

// =================================================================
//
// File: example8.cu
// Author(s):
// Description: This file contains the code that implements the
//				enumeration sort algorithm using CUDA.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
//Tiempo de ejecución paralelo:     0.0029 ms
//Tiempo de ejecución secuencial: 282.8298 ms
//Speed Up: 97,527.5172

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 10000
#define THREADS 256
#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))

__global__ void enum_sort(int* arr, int * c, int size) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int pos;

  while (tid < SIZE){
    pos=0;
    for(int i=0; i<size; i++){
      if(arr[i] < arr[tid] || arr[i] == arr[tid] && i < tid){
        pos++;
      }
    }
    c[pos]=arr[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main(int argc, char* argv[]) {
	int i, *a, *d_a, *c, *d_c;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("a", a);
  c = (int *) malloc(sizeof(int) * SIZE);

	cudaMalloc( (void**) &d_a, SIZE * sizeof(int) );
  cudaMalloc( (void**) &d_c, SIZE * sizeof(int) );

  printf("Starting...\n");
	ms = 0;
  cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	for (i = 1; i <= N; i++) {
		start_timer();
    enum_sort<<<BLOCKS, THREADS>>>(d_a, d_c, SIZE);
		ms += stop_timer();
	}

  cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c", c);
  printf("avg time = %.5lf ms\n", (ms / N));

	cudaFree(d_a);
  cudaFree(d_c);

	free(a);
  free(c);

  return 0;
}
