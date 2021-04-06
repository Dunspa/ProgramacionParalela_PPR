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

	// GPU phase
	t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for (int k = 0 ; k < niters ; k++) {
		//printf("CUDA kernel launch \n");
		// Tamaño del bloque (número de hebras por bloque)
	 	int threadsPerBlock = blocksize;
		// Tamaño del grid (número de bloques por grid)
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

	  	floyd_kernel_1d<<< blocksPerGrid, threadsPerBlock >>>(d_In_M, nverts, k);
	  	err = cudaGetLastError();

	  	if (err != cudaSuccess) {
	  		fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  		exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double tgpu1 = cpuSecond() - t1;

	// Tiempo en GPU con bloques unidimensionales
	cout << tgpu1 << " ";

	// Ganancia en velocidad
	cout << tcpu / tgpu1 << endl;

	for (int i = 0 ; i < nverts ; i++)
		for (int j = 0 ; j < nverts ; j++)
			if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0)
				cout << "Error (" << i << "," << j << ")   " 
					<< c_Out_M[i * nverts + j] << "..." 
					<< G.arista(i,j) << endl;
}
