#include "mpi.h"
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size, tama;
    MPI_Status estado;

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv); 
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) {
            cout << "No se ha especificado número de elementos, por defecto será "
                 << size * 100 << endl
                 << "Uso: <ejecutable> <cantidad>" << endl;
        }

        tama = size * 100;
    } else {
        tama = atoi(argv[1]);
        if (tama < size) {
            tama = size;
        } else {
            int i = 1, num = size;
            while (tama > num) {
                ++i;
                num = size * i;
            }

            if (tama != num) {
                if (rank == 0) {
                    cout << "Cantidad cambiada a " << num << endl;
                }

                tama = num;
            }
        }
    }

    // Creación y relleno de los vectores
    vector<long> VectorA, VectorB, VectorALocal;
    VectorA.resize(tama, 0);
    VectorB.resize(tama/size, 0);   // Subvector local
    VectorALocal.resize(tama/size, 0);
    if (rank == 0) {
        for (long i = 0 ; i < tama ; ++i) {
            VectorA[i] = i + 1; // Recibe valores 1, 2, 3, ..., tama
        }
    }

    // Repartimos los valores del vector A
    MPI_Scatter(&VectorA[0], tama/size, MPI_LONG, &VectorALocal[0], tama/size, MPI_LONG, 0, MPI_COMM_WORLD);

    // Damos valores al bloque local del vector B
    for (long i = 0 ; i < tama/size ; ++i) {
        VectorB[i] = (i + 1) * 10;
    }

    // Cálculo de la multiplicación escalar entre vectores
    long producto = 0;
    for (long i = 0 ; i < tama / size ; ++i) {
        producto += VectorALocal[i] * VectorB[i];
    }
    long total;

    // Reunimos los datos en un solo proceso
    MPI_Reduce(&producto, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Total: " << total << endl;
    }

    MPI_Finalize();

    return 0;
}