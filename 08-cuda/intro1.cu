// =================================================================
//
// File: intro1.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

//nvcc  intro1.cu
//./a.out

#include <stdio.h>
#include <cuda_runtime.h>

//definir hilo principal de ejecución __global__
__global__ void add(int *a, int *b, int *c) {
	//info en localidad de memoria a y b sumadas y guardar en la localidad de c
	//secuencia de trabajo que ejecutará el hilo
	*c = *a + *b;
}

int main(int argc, char* argv[]) {
	//variables locales que estarán en host
	int a, b, c;
	//copias de variables en device => apuntadores al bloque de memoria
	int *d_a, *d_b, *d_c;

	//cudaMalloc: pedir localidad de memoria
	//recibe localidad de memoria, donde quiero que lo coloque d_a
	//devuele el apuntador al apuntador
	cudaMalloc((void**) &d_a, sizeof(int));
	cudaMalloc((void**) &d_b, sizeof(int));
	cudaMalloc((void**) &d_c, sizeof(int));

	//después de pedir los bloques de memoria a la GPU,
	scanf("%i %i", &a, &b);

	//cudaMemcpy: pasar datos de forma local al GPU
	//destino es localidad del GPU, copiar bloque de memoria
	//referenciado por a, de Host a Dispositivo (RAM CPU -> RAM GPU)
	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	//Lanzar kernerl, quiero 1 hilo de trabajo, matriz 1 por 1
	add<<<1, 1>>>(d_a, d_b, d_c);

	//resultado es en memory de GPU, copiar de mem de GPU a la local
	cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("c = %i\n", c);

	//Liberar los bloques de memoria
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
