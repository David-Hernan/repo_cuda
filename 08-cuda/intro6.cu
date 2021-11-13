// =================================================================
//
// File: intro6.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

//Como una matriz aplanada

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 	1000000 //1e6
#define THREADS 256 //256 threads por bloeus
#define BLOCKS	MMAX(32, ((SIZE / THREADS) + 1)) //32 bloques min

__global__ void add(int *a, int *b, int *c) {
	//calcular posición de arrreglo
	//num bloque*tamaño bloque
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	//pregunto que i sea menor a SIZE para que no se exceda
	if (i < SIZE) {
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	//variables locales
	a = (int*) malloc(SIZE * sizeof(int));
	fill_array(a, SIZE);
	display_array("a", a);

	b = (int*) malloc(SIZE * sizeof(int));
	fill_array(b, SIZE);
	display_array("b", b);

	c = (int*) malloc(SIZE * sizeof(int));

	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));

	//muevo a device/GPU
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	//creo grupo de trabajo
	add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c);

	//regreso resultado
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c", c);

	//libero
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	free(c);
	free(b);
	free(a);

	return 0;
}
