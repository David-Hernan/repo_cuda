// =================================================================
//
// File: example7.cu
// Author(s):
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using CUDA.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "utils.h"

#define MAXIMUM 10000 //1e6 1000000
#define THREADS 256
//#define BLOCKS	MMIN(32, ((SIZE / THREADS) + 1))
#define BLOCKS	MMIN(32, ((MAXIMUM / THREADS) + 1))
// implement your code
__global__ void even(int* arr, int size) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);

  int j, prime;

  while (tid < MAXIMUM) {
    if(tid>=2){
      j=2;
			prime=1;
			while(j<tid){
				if(tid%j == 0){
					prime=0;
					arr[tid]=0;
					break;
				}
				j++;
			}
			if(prime == 1){
				arr[tid]=1;
			}
    }
		tid += blockDim.x * gridDim.x;
	}
}

int main(int argc, char* argv[]) {
	int i, j, *a, *d_a;
	double ms;

	a = new int[MAXIMUM + 1];

	cudaMalloc( (void**) &d_a, MAXIMUM * sizeof(int) );

  printf("Starting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
    cudaMemcpy(d_a, a, MAXIMUM * sizeof(int), cudaMemcpyHostToDevice);

		start_timer();

    for (j = 0; j <= MAXIMUM / 2; j++) {
      even<<<BLOCKS, THREADS>>>(d_a, MAXIMUM);
    }

		ms += stop_timer();
	}

  cudaMemcpy(a, d_a, MAXIMUM * sizeof(int), cudaMemcpyDeviceToHost);
  printf("Expanding the numbers that are prime to TOP_VALUE:\n");
  for (int i = 2; i < TOP_VALUE; i++) {
		if (a[i] == 1) {
			printf("%i ", i);
		}
	}
	//display_array("a", a);
  printf("avg time = %.5lf ms\n", (ms / N));

	cudaFree(d_a);

	free(a);

  return 0;
}
