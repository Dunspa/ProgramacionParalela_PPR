#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

// Kernel de advección lineal (memoria compartida)
__global__ void FD_kernel2(float * d_phi, float * d_phi_new, float cu, int n) { 
    int li = threadIdx.x + 1;   // índice local del vector en memoria compartida
	int gi = blockDim.x * blockIdx.x + threadIdx.x + 1; // índice global
    int lstart = 0;   
    int lend = BLOCKSIZE + 1;
	float result;
    __shared__ float s_phi[BLOCKSIZE + 2];

   	// Almacenar en memoria compartida
    if (gi < n + 2) {
		s_phi[li] = d_phi[gi];
	}  

	// Primera hebra del bloque
   	if (threadIdx.x == 0) {
		s_phi[lstart] = d_phi[gi - 1]; 
	}

	// Última hebra
   	if (threadIdx.x == BLOCKSIZE - 1)
	   	// Último bloque  
	  	if (gi >= n + 1)
	      	s_phi[(n + 2) % BLOCKSIZE] = d_phi[n + 2];
	  	else   
	      	s_phi[lend] = d_phi[gi + 1];

	// Esperar a que todas las hebras hayan llenado la memoria compartida
  	__syncthreads();
       
   	if (gi < n + 2) {
		// Lax-Friedrichs
		result = 0.5 * ((s_phi[li + 1] + s_phi[li - 1])
				- cu * (s_phi[li + 1] - s_phi[li - 1]));        
		d_phi_new[gi] = result;
    }
    
   	// Boundary Conditions
    if (gi == 1) {
		d_phi_new[0] = d_phi_new[1];
	}

    if (gi == n + 1) {
		d_phi_new[n + 2] = d_phi_new[n + 1];
	}
}

// Kernel que realiza el cálculo del vector B con memoria compartida
__global__ vectorial_shared(float * A, float * B, const int n) {

}

// Kernel que realiza el cálculo del vector B en memoria global
__global__ vectorial_global(float * A, float * B, const int n) {

}

// Kernel que calcula el valor máximo de entre los valores de un vector usando reduce
__global__ void maximo_reduce(float * B, float * max, const int n) {
	extern __shared__ int sdata[];	// Datos en memoria compartida

	int tid = threadIdx.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;	// Índice global de hebra

	// Almacenar vector en memoria compartida
	// Comprobar que la hebra del bloque puede operar en el vector
	// Si no, se le pone un valor demasiado bajo para que no pueda ser el máximo
	sdata[tid] = (i < n + 2) ? A[i] : -1000000.0f;
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
return float maximo_reduce_cpu(float * h_max) {
	// Terminar de computar el reduce en CPU
	// Se calcula el máximo de los encontrados en los distintos bloques
	float mx = -1000000.0f;
	for (int i = 0 ; i < blocksPerGrid ; i++) {
		mx = (longitudMaxima > h_max[i]) ? longitudMaxima : h_max[i];
	}

	return mx;
}

int main(int argc, char* argv[]) { 
	if (argc != 3) {
		cerr << "Sintaxis: " << argv[0] << " <tamaño vector>" << " <blocksize>" << endl;
		return(-1);
	}
	
	// Número de puntos
	const int n = atoi(argv[1]);
	// Tamaño de bloque
	const int blocksize = atoi(argv[2]);

	// Inicializar vector A
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

	// Inicializar vector B en GPU
	float * d_B = NULL;
	err = cudaMalloc((void **) &, n*sizeof(float)); 
	if (err != cudaSuccess) {
		cout << "ERROR DE RESERVA DE MEMORIA" << endl;
	}

	// Obtener información de GPU
    int devID;
    cudaDeviceProp props;
    cudaError_t err;
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
	
	double Tgpu1 = clock();
	Tgpu1 = (Tgpu1 - t1) / CLOCKS_PER_SEC;

	// Tiempo inicial
	t1 = clock();

	// Realizar operación vectorial en memoria global
	vectorial_global<<< blocksPerGrid, threadsPerBlock >>>(A, B, n);
	
	double Tgpu2 = clock();
	Tgpu2 = (Tgpu2 - t1) / CLOCKS_PER_SEC;

	// Calcular el valor máximo de todos los valores almacenados en B
	int * d_max; 
	cudaMalloc ((void **) &d_max, sizeof(float)*blocksPerGrid);
	maximo_reduce<<< blocksPerGrid, threadsPerBlock, smemSize >>>(B, d_max, n);	
	float * h_max = (float*) malloc(blocksPerGrid*sizeof(float));
	cudaMemcpy(h_max, d_max, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

	// Terminar reduce en CPU
	mx = maximo_reduce_cpu(h_max);

	// Mostrar tiempos de CPU, GPU SHARED, GPU GLOBAL, SPEEDUP SHARED, SPEEDUP GLOBAL
	cout << Tcpu << " " << Tgpu1 << " " << Tgpu2 << " " 
		 << Tcpu / Tgpu1 << " " << Tcpu / Tgpu2;

	return 0;
}





