#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
using namespace std;

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char *argv[]) {
    int **A, *x, *y;
    int n;
    double tInicio, tFin;

    // Leer parámetros
    if (argc < 2) {
        cout << "Número de parámetros incorrecto, es necesario indicar el tamaño del vector" << endl;
    } else {
        n = atoi(argv[1]);
    }

    // Reserva de vectores y matriz
    A = new int *[n];
    x = new int [n];

    A[0] = new int [n * n];
    for (unsigned int i = 1 ; i < n ; i++) {
        A[i] = A[i - 1] + n;
    }

    // Reservamos espacio para el resultado
    y = new int [n];

    // Rellenamos A y x con valores aleatorios
    srand(time(0));
    cout << "La matriz y el vector generados son " << endl;
    for (unsigned int i = 0 ; i < n ; i++) {
        for (unsigned int j = 0 ; j < n ; j++) {
            if (j == 0) {
                cout << "[";
            }

            A[i][j] = rand() % 1000;
            cout << A[i][j];

            if (j == n - 1) {
                cout << "]";
            } else {
                cout << " ";
            }
        }

        x[i] = rand() % 100;
        cout << "\t [" << x[i] << "]" << endl;
    }
    cout << endl;

    tInicio = cpuSecond();

    // Lo calculamos de forma secuencial
    for (unsigned int i = 0 ; i < n ; i++) {
        y[i] = 0;
        
        for (unsigned int j = 0 ; j < n ; j++) {
            y[i] += A[i][j] * x[j];
        }
    }

    tFin = cpuSecond();

    cout << "Tiempo empleado: " << tFin - tInicio << endl;

    delete [] x;
    delete [] y;
    delete [] A[0];
    delete [] A;
}