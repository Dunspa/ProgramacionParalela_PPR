#include "mpi.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size, respuesta;
    int rank_nuevo, size_nuevo;
    MPI_Comm comm;
    MPI_Status estado;

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv); 
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Crear nuevo comunicador a partir del global, agrupando por par/impar
    int parimpar = rank % 2;    // 1 (impar) o 0 (par)
    MPI_Comm_split(MPI_COMM_WORLD, parimpar, rank, &comm);
    // Obtener el nuevo rango asignado dentro de comm
    MPI_Comm_rank(comm, &rank_nuevo);
    // Obtener el nuevo número de procesos en el comunicador
    MPI_Comm_size(comm, &size_nuevo);

    // Realizar envíos y recepciones en el comunicador que les corresponda
    if (rank == 0 || rank == 1) {
        cout << "Soy el proceso " << rank << " GLOBAL y " << rank_nuevo 
             << " en COMM y envío mi rank global" << endl;

        // Enviar primer mensaje par e impar
        MPI_Send(&rank, 1, MPI_INT, rank_nuevo + 1, 0, comm);
    } else {
        // Recibir mensajes
        MPI_Recv(&respuesta, 1, MPI_INT, rank_nuevo - 1, 0, comm, &estado);

        cout << "Soy el proceso " << rank << " y he recibido " << respuesta << endl;

        if (rank_nuevo != size_nuevo - 1) {
            MPI_Send(&respuesta, 1, MPI_INT, rank_nuevo + 1, 0, comm);
        }
    }

    MPI_Finalize();

    if (rank == 0) {
        cout << "Fin del programa" << endl;
    }

    return 0;
}