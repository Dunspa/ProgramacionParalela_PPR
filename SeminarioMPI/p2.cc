#include "mpi.h"
#include <iostream>
#include <math.h>
using namespace std;

int main(int argc, char *argv[]) {
    int rank, size, n;
    double h, x, sum;
    double mypi, pi = 0.0;    // Valor local y global de PI
    double PI25 = 3.141592653589793238462643;
    bool valor_por_parametros = true;
    MPI_Status estado;

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv); 
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        cout << "Introduce la precisión del cálculo (subintervalos): ";
        cin >> n;
    }

    // El proceso 0 reparte al resto de procesos el número total de subintervalos
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int subintervalos = ceil(n / (double)size);

    // Comprobar que el número de iteraciones es correcto
    if (n <= 0) {
        MPI_Finalize();
        exit(0);
    } else {
        // Cálculo de PI
        h = 1.0 / (double) n;
        sum = 0.0;
        int istart = rank * subintervalos + 1;  
        int iend = subintervalos * (rank + 1);
        // Cambiar valor de iend si el proceso tiene menos subintervalos
        if (iend > n) {
            iend = iend - (iend % n);
        }

        if (istart > iend) {
            cout << "Proceso " << rank << " no calcula nada" << endl;
        } else {
            cout << "Proceso " << rank << " calcula desde el " << istart << " hasta el " << iend << endl;
        }
        
        for (int i = istart ; i <= iend ; ++i) {
            x = h * ((double)i - 0.5);
            sum += (4.0 / (1.0 + x*x));
        }
        mypi = h * sum;

        cout << "MI PI ES " << mypi << endl;

        // Todos los procesos comparen su valor de pi local
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Compartir la aproximación de PI con todos los procesos
        MPI_Bcast(&pi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Todos los procesos imprimen el mensaje
        cout << "El valor aproximado de PI es: " << pi << ", con un error de "
             << fabs(pi - PI25) << endl;
    }

    MPI_Finalize();

    return 0;
}