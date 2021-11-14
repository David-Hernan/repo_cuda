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
#define BLOCKS	MMIN(32, ((MAXIMUM / THREADS) + 1))

__global__ void GetPrime(int *array, int pos) {

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	int aux, b, i;
	while (tid < MAXIMUM) {
        // array[tid]=1;
        // aux=1;
        // for(i = 2; i <= sqrt((float)tid); i++){
        //         if(!(tid%i)){
        //             aux=0;
        //         }
		// 	}
        //     array[tid]=aux;
        //     tid += blockDim.x * gridDim.x;
        // }

			b=i/2;
			if(b<2){
				b=2;
			}
			aux=0;
			for(i = 2; i <= b; i++){
				if(array[tid-1] % i == 0){
					aux=0;
				}else{
					aux=1;
				}
			}
			array[tid]=aux;
            tid += blockDim.x * gridDim.x;
	}
}

int main(int argc, char* argv[]) {
	int i, *a, *d_a;
	double ms;

	a = (int*) malloc(sizeof(int) * MAXIMUM);
	fill_array(a, MAXIMUM);
	display_array("before", a);

    cudaMalloc( (void**) &d_a, MAXIMUM * sizeof(int) );
    cudaMemcpy(d_a, a, MAXIMUM * sizeof(int), cudaMemcpyHostToDevice);

    printf("\nStarting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
		start_timer();

        GetPrime<<<BLOCKS, THREADS>>> (d_a, MAXIMUM);

		ms += stop_timer();
	}

    cudaMemcpy(a, d_a, MAXIMUM * sizeof(int), cudaMemcpyDeviceToHost);


	for (i = 2; i < TOP_VALUE; i++) {
		if(a[i] == 1){
            printf("%i ",i);
        }
	}

    printf("\navg time = %.5lf ms\n", (ms / N));

    cudaFree(d_a);
	free(a);

    return 0;
}
