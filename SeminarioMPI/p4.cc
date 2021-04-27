#include "mpi.h"
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size;
    int rank_inverso, size_inverso;
    int rank_nuevo, size_nuevo;
    int a, b;
    int color;
    vector<int> vec, vecLocal;
    MPI_Status estado;
    MPI_Comm comm, comm_inverso;

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv); 
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        a = 2000;
        b = 1;
    } else {
        a = 0;
        b = 0;
    }

    color = rank % 2; // 1 o 0

    // Crear nuevo comunicador a partir del global por color y ordenados por rango
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm);

    // Crear nuevo comunicador a partir del global ordenados inversamente por rango
    MPI_Comm_split(MPI_COMM_WORLD, 0, -rank, &comm_inverso);

    // Obtener el nuevo rango asignado dentro de comm
    MPI_Comm_rank(comm, &rank_nuevo);
    MPI_Comm_rank(comm_inverso, &rank_inverso);
    // Obtener el nuevo número de procesos en el comunicador
    MPI_Comm_size(comm, &size_nuevo);
    MPI_Comm_size(comm_inverso, &size_inverso);

    // Vector local con un sólo elemento
    vecLocal.resize(1, 0);

    // Probamos a enviar datos por distintos comunicadores
    MPI_Bcast(&b, 1, MPI_INT, size - 1, comm_inverso);
    MPI_Bcast(&a, 1, MPI_INT, 0, comm);

    if (rank == 1) {
        // Inicializar vector en el proceso 1 GLOBAL (0 en los impares)
        vec.resize(size_nuevo, 0);
        for (int i = 0 ; i < size_nuevo ; ++i) {
            vec[i] = rand() % 1000;
        }
    }

    // Invocar Scatter del vector en comm pero solo para los impares de GLOBAL
    if (rank % 2 == 1) {
        MPI_Scatter(&vec[0], 1, MPI_INT, &vecLocal[0], 1, MPI_INT, 0, comm);
    }
    
    cout << "Soy el proceso " << rank << " de " << size << " dentro de MPI_COMM_WORLD"
         << endl << "\tMi rango en COMM_nuevo es " << rank_nuevo << " de " << size_nuevo
         << ", aquí he recibido el valor " << a << endl
         << "\tEn COMM_inverso mi rango es " << rank_inverso << " de " << size_inverso
         << " aquí he recibido el valor " << b << endl
         << "\tMi valor del vector es " << vecLocal[0] << endl << endl;

    MPI_Finalize();
    return 0;
}