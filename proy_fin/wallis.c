/*----------------------------------------------------------------
*
* Multiprocesadores: Proyecto final
* Fecha: 23-Nov-2021
* Autor: A01173130 David Hernán García Fernández
*
*--------------------------------------------------------------*/

// ==============================================================
// Descripción: Este archivo contiene el código que, de manera
//				secuencial, calcula una aproximación de PI mediante el
//				método del Producto de Wallis.
// ==============================================================

//Tiempo de ejecución: 19706.957 ms

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#define SIZE 1000000000 //1e9 1000000000

double sum_array(int *array, int size) {
	double result = 1.0;
  double val1 = 0.0;
  double val2 = 0.0;
  for (int i = 0; i < SIZE; i++) {
    val1 = (2.0*array[i])/((2.0*array[i])-1);
    val2 = (2.0*array[i])/((2.0*array[i])+1);
    result *= val1*val2;
  }
	return result;
}

int main(int argc, char* argv[]) {
	int i, *a;
	double ms, result;

	a = (int *) malloc(sizeof(int) * SIZE);
  for (i = 1; i < SIZE+1; i++) {
    a[i-1] = i;
  }

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		result = sum_array(a, SIZE);

		ms += stop_timer();
	}
  result*=2;
	printf("pi = %.16f para %d iteraciones.\n", result,SIZE);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	return 0;
}
