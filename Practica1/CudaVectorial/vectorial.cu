#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

// Kernel que realiza el cálculo del vector B con memoria compartida
__global__ void vectorial_shared(float * A, float * B, const int n) {
	extern __shared__ float sdata[];	// Datos en memoria compartida
	int i = threadIdx.x + blockDim.x * blockIdx.x;	// Índice global de hebra
	float Ai, Aim1, Aim2, Aip1, Aip2; 

	if (i >= 2 && i < n + 2) {
		// Almacenar A en memoria compartida
		sdata[i] = A[i];

		// Esperar a que todas las hebras terminen de llenar la memoria compartida
		__syncthreads();

		// Hacer la operación usando la memoria compartida
		// Cada hebra se ocupa de una posición del vector B
		Aim2 = sdata[i - 2];
		Aim1 = sdata[i - 1];  
		Ai = sdata[i];
		Aip1 = sdata[i + 1];  
		Aip2 = sdata[i + 2];
		B[i] = (pow(Aim2, 2) + 2.0 * pow(Aim1, 2) + pow(Ai, 2) 
		- 3.0 * pow(Aip1, 2) + 5.0 * pow(Aip2, 2)) / 24.0; 
	}
}

// Kernel que realiza el cálculo del vector B en memoria global
__global__ void vectorial_global(float * A, float * B, const int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;	// Índice global de hebra
	float Ai, Aim1, Aim2, Aip1, Aip2; 

	if (i >= 2 && i < n + 2) {
		// Hacer la operación usando la memoria global
		// Cada hebra se ocupa de una posición del vector B
		Aim2 = B[i - 2];
		Aim1 = B[i - 1];  
		Ai = B[i];
		Aip1 = B[i + 1];  
		Aip2 = B[i + 2];
		B[i] = (pow(Aim2, 5) + 2.0 * pow(Aim1, 5) + pow(Ai, 5) 
		- 3.0 * pow(Aip1, 5) + 5.0 * pow(Aip2, 5)) / 24.0; 
	}
}

// Kernel que calcula el valor máximo de entre los valores de un vector usando reduce
__global__ void maximo_reduce(float * B, float * max, const int n) {
	extern __shared__ float sdata[];	// Datos en memoria compartida

	int tid = threadIdx.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;	// Índice global de hebra

	// Almacenar vector en memoria compartida
	// Comprobar que la hebra del bloque puede operar en el vector
	// Si no, se le pone un valor demasiado bajo para que no pueda ser el máximo
	sdata[tid] = (i < n + 2) ? B[i] : -1000000.0f;
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

// Función para terminar de hacer el máximo con reduce en CPU
float maximo_reduce_cpu(float * h_max, int blocksPerGrid) {
	// Terminar de computar el reduce en CPU
	// Se calcula el máximo de los encontrados en los distintos bloques
	float mx = -1000000.0f;
	for (int i = 0 ; i < blocksPerGrid ; i++) {
		mx = (mx > h_max[i]) ? mx : h_max[i];
	}

	return mx;
}

int main(int argc, char* argv[]) { 
	if (argc != 3) {
		cerr << "Sintaxis: " << argv[0] << " <tamaño vector>" << " <blocksize>" << endl;
		return(-1);
	}
	
	// Gestionar errores cuda
	cudaError_t err;
	// Número de puntos
	const int n = atoi(argv[1]);
	// Tamaño de bloque
	const int blocksize = atoi(argv[2]);
	// Valor máximo
	float mx = 0.0f;

	// Inicializar vector A en CPU
	float * A = new float[n + 4];
	for (int i = 2 ; i < n + 2 ; i++) {
		A[i] = ((float) rand()) / (float) RAND_MAX;
	}
	A[0] = A[1] = A[n + 2] = A[n + 3] = 0.0;

	// Inicializar vector A en GPU
	float * d_A = NULL;
	err = cudaMalloc((void **) &d_A, (n + 4)*sizeof(float)); 
	if (err != cudaSuccess) {
		cout << "ERROR DE RESERVA DE MEMORIA" << endl;
	}
	err = cudaMemcpy(d_A, A, (n + 4)*sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	// Inicializar vector B en CPU
	float * B = new float[n];

	// Inicializar vector B en GPU
	float * d_B = NULL;
	err = cudaMalloc((void **) &d_B, n*sizeof(float)); 
	if (err != cudaSuccess) {
		cout << "ERROR DE RESERVA DE MEMORIA" << endl;
	}

	// Obtener información de GPU
    int devID;
    cudaDeviceProp props;
    err = cudaGetDevice(&devID);
    if (err != cudaSuccess) {
		cout << "ERROR" << endl;
	}  
    cudaGetDeviceProperties(&props, devID);
    //printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);
	
	// Fase CPU
	double t1cpu = clock();   

	float Ai, Aim1, Aim2, Aip1, Aip2;  
	for (int i = 2 ; i < n + 2 ; i++){ 
		// Operación vectorial
		const int iB = i - 2;
		Aim2 = A[i - 2];
		Aim1 = A[i - 1];  
		Ai = A[i];
		Aip1 = A[i + 1];  
		Aip2 = A[i + 2];
		B[iB] = (pow(Aim2, 2) + 2.0 * pow(Aim1, 2) + pow(Ai, 2) 
		- 3.0 * pow(Aip1, 2) + 5.0 * pow(Aip2, 2)) / 24.0; 

		// Cálculo del máximo
		mx = (iB == 0) ? B[0] : max(B[iB], mx);
	}
	
	double Tcpu = clock();
	Tcpu = (Tcpu - t1cpu) / CLOCKS_PER_SEC;
	
	// Fase GPU  
	// Tamaño del bloque (número de hebras por bloque)
	int threadsPerBlock = blocksize;
	// Tamaño del grid (número de bloques por grid)
	int blocksPerGrid = ceil((float) (n + 2) / threadsPerBlock);
	// Tamaño de la memoria compartida
	int smemSize = threadsPerBlock * sizeof(float);

	// Tiempo inicial
	double t1 = clock();

	// Realizar operación vectorial en memoria compartida
	vectorial_shared<<< blocksPerGrid, threadsPerBlock, smemSize >>>(A, B, n);

	cudaDeviceSynchronize();
	
	double Tgpu1 = clock();
	Tgpu1 = (Tgpu1 - t1) / CLOCKS_PER_SEC;

	// Tiempo inicial
	t1 = clock();

	// Realizar operación vectorial en memoria global
	vectorial_global<<< blocksPerGrid, threadsPerBlock >>>(A, B, n);

	cudaDeviceSynchronize();
	
	double Tgpu2 = clock();
	Tgpu2 = (Tgpu2 - t1) / CLOCKS_PER_SEC;

	// Calcular el valor máximo de todos los valores almacenados en B
	float * d_max; 
	cudaMalloc ((void **) &d_max, sizeof(float)*blocksPerGrid);
	maximo_reduce<<< blocksPerGrid, threadsPerBlock, smemSize >>>(B, d_max, n);	
	float * h_max = (float*) malloc(blocksPerGrid*sizeof(float));
	cudaMemcpy(h_max, d_max, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

	// Terminar reduce en CPU
	mx = maximo_reduce_cpu(h_max, blocksPerGrid);

	// Mostrar tiempos de CPU, GPU SHARED, GPU GLOBAL, SPEEDUP SHARED, SPEEDUP GLOBAL
	printf("%i %.7f %.7f %.7f %.7f %.7f\n", blocksize, Tcpu, Tgpu1, Tgpu2, Tcpu / Tgpu1, Tcpu / Tgpu2);

	return 0;
}





