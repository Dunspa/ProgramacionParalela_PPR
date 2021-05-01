#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size, n;
    int **A, *x, *y, *miBloque, *comprueba, *subFinal;
    double tInicio, tFin;
    MPI_Status estado;

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv); 
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Leer parámetros
    if (argc < 2) {
        cout << "Número de parámetros incorrecto, es necesario indicar el tamaño del vector" << endl;
    } else {
        n = atoi(argv[1]);
        
        // Comprobar si n es un múltiplo del número de procesos y es mucho mayor
        // Como mínimo debe ser mayor que el doble del número de procesos
        if (n % size == 0 && n >= 2 * size) {
            if (rank == 0) {
                cout << "Tamaño del vector: " << n << endl << endl;
            }
        } else {
            if (rank == 0) {
                cout << "El tamaño del vector indicado no es correcto." 
                     << " Debe ser un múltiplo del número de procesos (" << size
                     << ") y además debe ser mayor que el doble de este" << endl;
            }
            
            // Transformar valor n a uno válido
            do {
                n++;
            } while (n % size != 0);

            if (rank == 0) {
                cout << "Se ha transformado el valor para que sea correcto" << endl
                     << "Tamaño del vector: " << n << endl << endl;
            }
        }
    }

    // Reserva de vectores y matriz
    A = new int *[n];
    x = new int [n];

    if (rank == 0) {
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

        // Reservamos espacio para la comprobación
        comprueba = new int [n];

        // Lo calculamos de forma secuencial
        for (unsigned int i = 0 ; i < n ; i++) {
            comprueba[i] = 0;
            
            for (unsigned int j = 0 ; j < n ; j++) {
                comprueba[i] += A[i][j] * x[j];
            }
        }
    }

    // Número de elementos y filas de la matriz que tiene cada proceso
    int nfilas = n / size;
    int nelementos = (n * n) / size;

    // Reservamos espacio para el bloque de filas local de cada proceso
    miBloque = new int [nelementos];
    // Reservamos espacio para los resultados que calcula cada proceso
    // Uno por fila que tiene
    subFinal = new int [nfilas];

    // Repartimos n/p filas por cada proceso
    MPI_Scatter(A[0], nelementos, MPI_INT, miBloque, nelementos, MPI_INT, 0, MPI_COMM_WORLD);

    // Compartimos el vector entre todos los procesos
    MPI_Bcast(x, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Barrera para asegurar que todos los procesos comiencen a la vez
    MPI_Barrier(MPI_COMM_WORLD);
    tInicio = MPI_Wtime();

    // Se tiene más de un subvector final, se calcula cada vez un resultado
    for (int i = 0 ; i < nfilas ; ++i) {
        subFinal[i] = 0;

        for (int j = 0 ; j < n ; j++) {
            subFinal[i] += miBloque[j + (n * i)] * x[j];
        }

        cout << subFinal[i] << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tFin = MPI_Wtime();

    // Recogemos los escalares de la multiplicación en un vector
    // Se hace en el mismo orden que el scatter
    MPI_Gather(&subFinal[0], 2, MPI_INT, y, 2, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0) {
        unsigned int errores = 0;

        cout << "El resultado obtenido y el esperado son: " << endl;
        for (unsigned int i = 0 ; i < n ; i++) {
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
    delete [] miBloque;

    return 0;
}