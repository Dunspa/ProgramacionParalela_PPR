#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size;
    int **A, *x, *y, *miFila, *comprueba;
    double tInicio, tFin;
    MPI_Status estado;

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv); 
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Reserva de vectores y matriz
    A = new int *[size];
    x = new int [size];

    if (rank == 0) {
        A[0] = new int [size * size];
        for (unsigned int i = 1 ; i < size ; i++) {
            A[i] = A[i - 1] + size;
        }

        // Reservamos espacio para el resultado
        y = new int [size];

        // Rellenamos A y x con valores aleatorios
        srand(time(0));
        cout << "La matriz y el vector generados son " << endl;
        for (unsigned int i = 0 ; i < size ; i++) {
            for (unsigned int j = 0 ; j < size ; j++) {
                if (j == 0) {
                    cout << "[";
                }

                A[i][j] = rand() % 1000;
                cout << A[i][j];

                if (j == size - 1) {
                    cout << "]";
                } else {
                    cout << " ";
                }
            }

            x[i] = rand() % 100;
            cout << "\t [" << x[i] << "]" << endl;
        }
        cout << endl;

        // Reservamos espacio para la comprobación
        comprueba = new int [size];

        // Lo calculamos de forma secuencial
        for (unsigned int i = 0 ; i < size ; i++) {
            comprueba[i] = 0;
            
            for (unsigned int j = 0 ; j < size ; j++) {
                comprueba[i] += A[i][j] * x[j];
            }
        }
    }

    // Reservamos espacio para la fila local de cada proceso
    miFila = new int [size];

    // Repartimos una fila por cada proceso
    MPI_Scatter(A[0], size, MPI_INT, miFila, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compartimos el vector entre todos los procesos
    MPI_Bcast(x, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Barrera para asegurar que todos los procesos comiencen a la vez
    MPI_Barrier(MPI_COMM_WORLD);
    tInicio = MPI_Wtime();

    int subFinal = 0;
    for (unsigned int i = 0 ; i < size ; i++) {
        subFinal += miFila[i] * x[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tFin = MPI_Wtime();

    // Recogemos los escalares de la multiplicación en un vector
    // Se hace en el mismo orden que el scatter
    MPI_Gather(&subFinal, 1, MPI_INT, y, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0) {
        unsigned int errores = 0;

        cout << "El resultado obtenido y el esperado son: " << endl;
        for (unsigned int i = 0 ; i < size ; i++) {
            cout << "\t" << y[i] << "\t|\t" << comprueba[i] << endl;
            if (comprueba[i] != y[i]) {
                errores++;
            }
        }

        delete [] y;
        delete [] comprueba;
        delete [] A[0];

        if (errores) {
            cout << "Hubo " << errores << " errores." << endl;
        } else {
            cout << "No hubo errores" << endl;
            cout << "El tiempo tardado ha sido " << tFin - tInicio << " segundos" << endl;
        }
    }

    delete [] x;
    delete [] A;
    delete [] miFila;
}