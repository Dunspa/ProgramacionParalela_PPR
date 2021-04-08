#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

// Blocksize
#define BLOCKSIZE 64
// Number of mesh points
int n = 60000;

int index(int i) {
	return i + 1;
}

// Intercambiar dos punteros a float
void swap_pointers(float * *a, float * *b) {
	float * tmp = *a;
	*a = *b;
	*b = tmp;
}

// Kernel de advección lineal (memoria global)
__global__ void FD_kernel1(float * d_phi, float * d_phi_new, float cu, int n) { 
	int i=threadIdx.x+blockDim.x*blockIdx.x+1;

    // Inner point update
    if (i<n+2) {
		d_phi_new[i] = 0.5 * ((d_phi[i + 1] + d_phi[i - 1]) 
		- cu * (d_phi[i + 1] - d_phi[i - 1])); 
	}        

    // Boundary Conditions
    if (i==1) {
		d_phi_new[0] = d_phi_new[1];
	}

    if (i==n+1) {
		d_phi_new[n + 2] = d_phi_new[n + 1];
	}
}

  

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

int main(int argc, char* argv[]) { 
	// Obtener información de GPU
    int devID;
    cudaDeviceProp props;
    cudaError_t err;
    err = cudaGetDevice(&devID);
    if (err != cudaSuccess) {
		cout << "ERROR" << endl;
	}  
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);
  

	cout << "Introduce el número de puntos (1000-200000)" << endl;
	cin >> n;

	// Tamaño del dominio (periódico) 
	float l = 10.0;
	// Grid
	float dx = l / n;
	// Velocidad de advección
	float u = 1.0;
	// Tiempo
	float dt = 0.8 * u * dx;
	float tend = 2.5;
	// Número de Courant
	float cu = u * dt / dx;
	
	// Número de pasos a tomar
	int nsteps = (int) ceil(tend / dt);
	
	cout << "dx=" << dx << "...  dt= " << dt << "...Courant= " << cu << endl;
	cout << endl;
	cout << "Número de pasos de tiempo=" << nsteps << endl;

	float * phi = new float[n + 3];
	float * phi_new = new float[n + 3];
	float * phi_GPU = new float[n + 3];
	float xx[n + 1];

	for (int i = 0 ; i <= n ; i++) {
		xx[i] = -5.0 + i * dx; 
	}
	
	// Valores iniciales para phi -> Gausiana
	for (int i = 0 ; i <= n ; i++) {
		// Gausiana
		phi[index(i)] = (1.0 / (2.0 * M_PI * 0.16)) 
		* exp(-0.5 * (pow((xx[i] - 0.5), 2) / 0.01));
	}
	
	// Fase GPU  
	int size = (n + 3) * sizeof(float);
	
	// Crear memoria en GPU
	float * d_phi = NULL;
	err = cudaMalloc((void **) &d_phi, size); 
	if (err != cudaSuccess) {
		cout << "ERROR DE RESERVA DE MEMORIA" << endl;
	}

	float * d_phi_new = NULL;
	err = cudaMalloc((void **) &d_phi_new, size); 
	if (err != cudaSuccess) {
		cout << "ERROR DE RESERVA DE MEMORIA" << endl;
	}

	// Tiempo inicial
	double t1=clock();

	// Boundary Conditions
	phi[index(-1)] = phi[index(0)];
	phi[index(n+1)] = phi[index(n)];

	// Copiar valores de phi a memoria de gpu
	err = cudaMemcpy(d_phi, phi, size, cudaMemcpyHostToDevice);

	if (err!=cudaSuccess) {
		cout << "ERROR AL COPIAR EN GPU" << endl;
	}

	// Iteración de paso de tiempo
	for (int k = 0 ; k < nsteps ; k++) {       
		int blocksPerGrid = (int) ceil((float)(n + 1) / BLOCKSIZE);

		FD_kernel2<<< blocksPerGrid, BLOCKSIZE >>> (d_phi, d_phi_new, cu, n);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel! %d \n", err);
			exit(EXIT_FAILURE);
		}

		swap_pointers (&d_phi, &d_phi_new); 
	}

	cudaMemcpy(phi_GPU, d_phi, size, cudaMemcpyDeviceToHost);

	double Tgpu=clock();
	Tgpu = (Tgpu - t1) / CLOCKS_PER_SEC;

	// Fase CPU
	double t1cpu = clock();   

	for (int k = 0 ; k < nsteps ; k++) { 
		// Boundary Conditions
		phi[index(-1)] = phi[index(0)];
		phi[index(n + 1)] = phi[index(n)];

		for(int i = 0 ; i <= n ; i++) {
			float phi_i = phi[index(i)];
			float phi_ip1 = phi[index(i + 1)];
			float phi_im1 = phi[index(i - 1)];

			// Lax-Friedrichs
			phi_new[index(i)] = 0.5 * ((phi_ip1 + phi_im1) - cu * (phi_ip1 - phi_im1));
		}	    
		
		swap_pointers (&phi,&phi_new);
	}
	
	
	double Tcpu = clock();
	Tcpu = (Tcpu - t1cpu) / CLOCKS_PER_SEC;


	cout << endl;
	cout << "Tiempo GPU= " << Tgpu << endl << endl;
	cout << "Tiempo CPU= " << Tcpu << endl << endl;

	// Comparación entre GPU y CPU
	// Comprobación de errores 
	int passed = 1;
	int i = 0;
	while (passed && i < n) {
		double diff = fabs((double)phi_GPU[index(i)] - (double)phi[index(i)]);
		if (diff > 1.0e-5) { 
			passed = 0; 
			cout << "DIFF= " << diff << endl;
		}

		i++;
	} 
		
	if (passed) 
		cout << "TEST CORRECTO" << endl;   
	else 
		cout << "ERROR EN EL TEST" << endl;
		
	cout << endl; 
	cout << "Speedup (T_CPU/T_GPU)= " << Tcpu / Tgpu << endl;
	
	return 0;
}





