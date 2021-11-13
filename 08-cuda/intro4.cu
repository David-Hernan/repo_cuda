// =================================================================
//
// File: intro4.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

//este código hace C[i]=A[i]+B[i]

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

//número de threads por bloque
#define SIZE 512

//instrucción a un hilo
__global__ void add(int *a, int *b, int *c) {
	//n renglones, una columna
	//block para renglones, thread para columnas
	//acá lo veo como n renglones, 1 columna
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(int argc, char* argv[]) {
	//genero arreglo locales
	int *a, *b, *c;
	//arreglo en device
	int *d_a, *d_b, *d_c;

	//arreglo 512 localidades
	a = (int*) malloc(SIZE * sizeof(int));
	//llenarlos
	fill_array(a, SIZE);
	//desplegarlos
	display_array("a", a);

	//igual creo arreglo B
	b = (int*) malloc(SIZE * sizeof(int));
	fill_array(b, SIZE);
	display_array("b", b);

	//creo array para resultados
	c = (int*) malloc(SIZE * sizeof(int));

	//lo anterior fue en CPU, esto en GPU
	//reserva bloques memoroa en GPU
	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));

	//transferir a GPU, c es resultado de suma así que no se transfiere
	//host CPU; device GPU
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	//SIZE renglones, 1 columnas
	//le indico que trabaje con los 3 bloques de memory d_a, d_b, d_c
	add<<<SIZE, 1>>>(d_a, d_b, d_c);

	//resultado está en c de GPU, ahora lo traigo al CPU
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	//mostrar resultado
	display_array("c", c);

	//Libero bloques de memoria en GPU y CPU
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	free(c);
	free(b);
	free(a);

	return 0;
}
