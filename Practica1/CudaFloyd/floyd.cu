#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

using namespace std;

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

// Kernel para resolver el algoritmo de Floyd con bloques unidimensionales
__global__ void floyd_kernel_1d(int * M, const int nverts, const int k) {
	int ij = threadIdx.x + blockDim.x * blockIdx.x;
    const int i = ij / nverts;
    const int j = ij - i * nverts;

    if (i < nverts && j < nverts) {
		int Mij = M[ij];

		if (i != j && i != k && j != k) { // Evitar los 0 de la matriz (evitar el mismo vertice)
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
  	}
}

// Kernel para resolver el algoritmo de Floyd con bloques bidimensionales
__global__ void floyd_kernel_2d(int * M, const int nverts, const int k) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;	// Índice de filas de hebra
	int i = blockIdx.y * blockDim.y + threadIdx.y; 	// Índice de columnas de hebra

	int ij = i * nverts + j;	// Índice global del elemento en la matriz
	// Para poder acceder a los valores dentro de la matriz por fila y columna
	int i2 = ij / nverts;		// Fila correspondiente al índice global
	int j2 = ij - i2 * nverts; 	// Columna correspondiente al índice global

    if (i < nverts && j < nverts) {
		int Mij = M[i2 * nverts + j2];

		// Evitar los 0 de la matriz (evitar el mismo vertice)
		if (i != j && i != k && j != k) {
			int Mikj = M[i2 * nverts + k] + M[k * nverts + j2];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
  	}
}

// Kernel para calcular la longitud del camino de mayor longitud dentro de los caminos
// más cortos encontrados usando anteriormente el algoritmo de Floyd
// Se realiza mediante reducción
__global__ void mayor_longitud_reduce(int * M, int * max, const int nverts) {
	extern __shared__ int sdata[];	// Datos en memoria compartida

	int tid = threadIdx.x;
	int ij = threadIdx.x + blockDim.x * blockIdx.x;	// Índice global de hebra
    const int i = ij / nverts;
    const int j = ij - i * nverts;

	// Comprobar que la hebra del bloque puede operar en la matriz y obtener su distancia
	// Si no, se le pone un valor demasiado bajo para que no pueda ser el máximo
	sdata[tid] = (i < nverts && j < nverts && M[i * nverts + j] != INF) ? M[i * nverts + j] : -1000000;
	// Esperar a que acaben todas las hebras para terminar de llenar la memoria compartida
	__syncthreads();

	// Hacer la reducción en memoria compartida
	// Se hace una división por dos más eficiente mediante un desplazamiento de bits
	for (int s = blockDim.x / 2 ; s > 0 ; s >>= 1) {
		// Comprobar que la hebra actual está dentro de la mitad que está operando
		if (tid < s) {
			// Calcular máximo en la posición que le corresponde a la hebra actual
			if (sdata[tid] < sdata[tid + s]) {
				sdata[tid] = sdata[tid + s];
			}
		}

		// Esperar a que acaben todas las hebras para poder decidir el máximo del bloque
		__syncthreads();
	}

	// La primera hebra se ocupa de escribir resultado de este bloque en la memoria local
	if (tid == 0) {
		max[blockIdx.x] = sdata[0];
	}
}

int main (int argc, char *argv[]) {
	if (argc != 3) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << " <blocksize>" << endl;
		return(-1);
	}

	// Tamaño de bloque
	const int blocksize = atoi(argv[2]);
	
	// Obtener la información de la GPU
	//int devID;
	//cudaDeviceProp props;
	cudaError_t err;
	/*err = cudaGetDevice(&devID);
  	if(err != cudaSuccess) {
		cout << "ERROR AL OBTENER LA INFORMACIÓN DE LA GPU" << endl;
	}
	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);*/

	// Leer el grafo del archivo 
	Graph G;
	G.lee(argv[1]);

	//cout << "El Grafo de entrada es:" << endl;
	//G.imprime();

	const int nverts = G.getVertices();
	cout << nverts << " ";
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	int * c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	// CPU phase
	double t1 = cpuSecond();

	// BUCLE PRINCIPAL DEL ALGORITMO
	int inj, in, kn;
	for (int k = 0 ; k < niters ; k++) {
        kn = k * nverts;

		for (int i = 0 ; i < nverts ; i++) {
			in = i * nverts;

			for (int j = 0; j < nverts; j++) {
				if (i!=j && i!=k && j!=k){
					inj = in + j;
					A[inj] = min(A[in+k] + A[kn+j], A[inj]);
				}
			}
		}
	}

  	double tcpu = cpuSecond() - t1;
	
	// Tiempo en CPU
  	cout << tcpu << " ";

	// GPU phase 1
	t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for (int k = 0 ; k < niters ; k++) {
		// Tamaño del bloque (número de hebras por bloque)
	 	int threadsPerBlock = blocksize * blocksize;
		// Tamaño del grid (número de bloques por grid)
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

	  	floyd_kernel_1d<<< blocksPerGrid, threadsPerBlock >>>(d_In_M, nverts, k);
	  	err = cudaGetLastError();

	  	if (err != cudaSuccess) {
	  		fprintf(stderr, "Failed to launch kernel 1! ERROR= %d\n",err);
	  		exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double tgpu1 = cpuSecond() - t1;

	// Tiempo en GPU con bloques unidimensionales
	cout << tgpu1 << " ";

	// GPU phase 2
	t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for (int k = 0 ; k < niters ; k++) {
		// Tamaño del bloque (número de hebras por bloque)
		dim3 threadsPerBlock(blocksize, blocksize);
		// Tamaño del grid (número de bloques por grid)
		int tamx = ceil((float) (nverts) / threadsPerBlock.x);
		int tamy = ceil((float) (nverts) / threadsPerBlock.y);
		dim3 blocksPerGrid(tamx, tamy);

	  	floyd_kernel_2d<<< blocksPerGrid, threadsPerBlock >>>(d_In_M, nverts, k);
	  	err = cudaGetLastError();

	  	if (err != cudaSuccess) {
	  		fprintf(stderr, "Failed to launch kernel 2! ERROR= %d\n",err);
	  		exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double tgpu2 = cpuSecond() - t1;

	// Tiempo en GPU con bloques bidimensionales
	cout << tgpu2 << " ";

	// Ganancia en velocidad de ambas versiones GPU con respecto a la monohebra
	cout << tcpu / tgpu1 << " " << tcpu / tgpu2 << endl;

	// Tamaño del bloque (número de hebras por bloque)
	int threadsPerBlock = blocksize * blocksize;
	// Tamaño del grid (número de bloques por grid)
	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
	// Tamaño de la memoria compartida
	int smemSize = threadsPerBlock * sizeof(int);
	// Longitud del camino de mayor longitud
	int * d_max; 
	cudaMalloc ((void **) &d_max, sizeof(int)*blocksPerGrid);

	// Calcular la longitud del camino de mayor longitud
	mayor_longitud_reduce<<< blocksPerGrid, threadsPerBlock, smemSize >>>(d_In_M, d_max, nverts);
	
	// Terminar de computar el reduce en CPU
	// Se calcula el máximo de los encontrados en los distintos bloques
	int * h_max = (int*) malloc(blocksPerGrid*sizeof(int));
	cudaMemcpy(h_max, d_max, blocksPerGrid*sizeof(int), cudaMemcpyDeviceToHost);
	int longitudMaxima = -1000000;
	for (int i = 0 ; i < blocksPerGrid ; i++) {
		longitudMaxima = (longitudMaxima > h_max[i]) ? longitudMaxima : h_max[i];
	}

	//cout << "LONGITUD DEL CAMINO MAYOR = " << longitudMaxima << endl;

	for (int i = 0 ; i < nverts ; i++)
		for (int j = 0 ; j < nverts ; j++)
			if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0)
				cout << "Error (" << i << "," << j << ")   " 
					<< c_Out_M[i * nverts + j] << "..." 
					<< G.arista(i,j) << endl;

	// Liberar toda la memoria
	free(c_Out_M);
	free(A);
	cudaFree(d_In_M);
}
