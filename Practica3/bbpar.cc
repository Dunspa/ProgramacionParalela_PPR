/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Paralelo                    */
/*                      Jose Luis Gallego Peña                          */
/* ******************************************************************** */
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "libbb.h"

using namespace std;

unsigned int NCIUDADES;
int rank, size, tag;
MPI_Status estado;

// Algoritmo de equilibrado de carga con detección de fin
void Equilibrar_Carga(tPila* pila, bool* activo) {
    // El proceso no tiene nodos para trabajar: pedir nodos a otros procesos
    if (pila.vacia()) {
        // Enviar mensaje de petición de trabajo
        MPI_Send(rank, 1, MPI_INT, siguiente, 0, MPI_COMM_WORLD);

        // Esperar a conseguir trabajo
        while (pila.vacia() && activo) {
            // Sondear respuesta
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);

            switch (estado.MPI_TAG) {
                // Petición de trabajo
                case PETICION:
                    // Recibir mensaje de petición de trabajo
                    MPI_Recv();

                    if (estado.MPI_SOURCE == rank) {  // Petición devuelta

                    } else {  // Petición de otro proceso
                        // Pasar petición de trabajo al siguiente proceso
                        MPI_Send();
                    }

                    break;

                // Resultado de una petición de trabajo
                case NODOS:
                    // Recibir nodos del proceso donante
                    int numnodos;
                    MPI_Get_count(&estado, MPI_INT, &numnodos);
                    MPI_Recv(pila.nodos, , MPI_INT, estado.MPI_SOURCE, MPI_COMM_WORLD, &estado);

                    // Almacenar nodos recibidos en la pila

                    break;

                case TOKEN:
                    break;

                case FIN:
                    break;
            }
        }
    }

    // El proceso tiene nodos para trabajar
    if (activo) {
        // Sondear si hay mensajes pendientes de otros procesos
        while (MPI_Iprobe()) {
            // Recibir mensaje de petición de trabajo
            MPI_Recv();
        }
    }
}

int main(int argc, char** argv) {
    switch (argc) {
        case 3:
            NCIUDADES = atoi(argv[1]);
            break;

        default:
            cerr << "La sintaxis es: bbpar <tamaño> <archivo>" << endl;
            exit(1);
            break;
    }

    // Inicialización del entorno MPI
    MPI_Init(&argc, &argv);
    // Obtenemos el número de procesos en el comunicador global
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Obtenemos la identificación de nuestro proceso en el comunicador global
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
    tNodo nodo,    // nodo a explorar
        lnodo,     // hijo izquierdo
        rnodo,     // hijo derecho
        solucion;  // mejor solucion
    bool activo,   // condicion de fin
        nueva_U;   // hay nuevo valor de cota superior
    int U;         // valor de cota superior
    int iteraciones = 0;
    tPila pila;  // pila de nodos a explorar

    U = INFINITO;  // inicializa cota superior

    // Anillo de procesos
    anterior = (rank - 1 + size) % size;
    siguiente = (rank + 1) % size;

    // Proceso 0 lee la matriz del problema inicial
    if (rank == 0) {
        // El proceso 0 empieza teniendo el token
        token_presente = true;

        InicNodo(&nodo);            // inicializa estructura nodo
        LeerMatriz(argv[2], tsp0);  // lee matriz de fichero
    }

    // Difusión de matriz del problema inicial al resto de procesos
    MPI_Bcast(tsp0[0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    activo = !Inconsistente(tsp0);

    // Primer reparto de carga
    if (rank != 0) {
        // El token lo tiene el proceso 0
        token_presente = false;

        // Realizar peticiones de trabajo
        Equilibrar_Carga(&pila, &activo);

        // No se cumple la condición de fin
        if (activo) {
            pila.pop(&nodo);
        }
    }

    double t = MPI_Wtime();

    // ciclo del Branch&Bound
    while (activo) {
        Ramifica(&nodo, &lnodo, &rnodo, tsp0);
        nueva_U = false;

        if (Solucion(&rnodo)) {
            if (rnodo.ci() < U) {  // se ha encontrado una solucion mejor
                U = rnodo.ci();
                nueva_U = true;
                CopiaNodo(&rnodo, &solucion);
            }
        } else {                   //  no es un nodo solucion
            if (rnodo.ci() < U) {  //  cota inferior menor que cota superior
                if (!pila.push(rnodo)) {
                    printf("Error: pila agotada\n");
                    liberarMatriz(tsp0);
                    exit(1);
                }
            }
        }

        if (Solucion(&lnodo)) {
            if (lnodo.ci() < U) {  // se ha encontrado una solucion mejor
                U = lnodo.ci();
                nueva_U = true;
                CopiaNodo(&lnodo, &solucion);
            }
        } else {                   // no es nodo solucion
            if (lnodo.ci() < U) {  // cota inferior menor que cota superior
                if (!pila.push(lnodo)) {
                    printf("Error: pila agotada\n");
                    liberarMatriz(tsp0);
                    exit(1);
                }
            }
        }

        // Se ha encontrado una nueva cota superior
        if (nueva_U) pila.acotar(U);

        Equilibrar_Carga(&pila, &activo);
        if (activo) {
            pila.pop(nodo);
        }

        iteraciones++;
    }

    t = MPI_Wtime() - t;

    // Obtener iteraciones totales
    // MPI_Reduce();

    MPI_Finalize();

    cout << "Solucion: \n" << cout;
    EscribeNodo(&solucion);
    cout << "Tiempo gastado = " << t << endl;
    cout << "Numero de iteraciones = " << iteraciones << endl << endl;

    liberarMatriz(tsp0);
}
