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
int rank, size, tag, flag;
int p_trabajo;  // Proceso al que enviar el trabajo
tPila pila2;    // pila de nodos a enviar
MPI_Status estado_mensaje;

// No ceder nodos si el tamaño de la pila está por debajo del umbral
const int UMBRAL_PILA = 2;

// Algoritmo de equilibrado de carga con detección de fin
void Equilibrar_Carga(tPila& pila, bool& activo) {
    // El proceso no tiene nodos para trabajar: pedir nodos a otros procesos
    if (pila.vacia()) {
        // Enviar mensaje de petición de trabajo
        MPI_Send(rank, 1, MPI_INT, siguiente, PETICION, MPI_COMM_WORLD);

        // Esperar a conseguir trabajo
        while (pila.vacia() && activo) {
            // Sondear mensajes
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado_mensaje);

            switch (estado_mensaje.MPI_TAG) {
                // Petición de trabajo
                case PETICION:
                    // Proceso en estado activo
                    estado = ACTIVO;

                    // Recibir mensaje de petición de trabajo
                    MPI_Recv(&p_trabajo, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, MPI_COMM_WORLD, &estado_mensaje);

                    if (estado_mensaje.MPI_SOURCE == rank) {  // Petición devuelta
                        estado = PASIVO;

                        // Reiniciar detección de fin
                        if (token_presente) {
                            MPI_Send(BLANCO, 1, MPI_INT, anterior, TOKEN, MPI_COMM_WORLD);
                        }
                    } else {  // Petición de otro proceso
                        
                    }

                    // Pasar petición de trabajo al siguiente proceso
                    MPI_Send(p_trabajo, 1, MPI_INT, siguiente, PETICION, MPI_COMM_WORLD);

                    break;

                // Resultado de una petición de trabajo
                case NODOS:
                    // Recibir nodos del proceso donante
                    int numnodos;
                    MPI_Get_count(&estado_mensaje, MPI_INT, &numnodos);
                    MPI_Recv(pila.nodos, numnodos, MPI_INT, estado_mensaje.MPI_SOURCE, NODOS, MPI_COMM_WORLD, &estado_mensaje);

                    break;

                // Se recibe el token
                case TOKEN:
                    token_presente = true;
                    MPI_Recv(&color_token, 1, MPI_INT, estado_mensaje.MPI_SOURCE, TOKEN, MPI_COMM_WORLD, &estado_mensaje);

                    // Si el proceso es pasivo, se envía el token al proceso anterior, si no, lo mantiene
                    if (estado == PASIVO) {
                        if (rank == 0 && color == BLANCO && color_token == BLANCO) {    // Terminación detectada
                            activo = false;
                        } else {
                            // Cambiar el color al token
                            if (rank == 0) {
                                color_token = BLANCO;
                            } else if (color == NEGRO) {
                                color_token = NEGRO;
                            }

                            token_presente = false;
                            MPI_Send(color_token, 1, MPI_INT, anterior, TOKEN, MPI_COMM_WORLD);
                        }
                    }

                    break;

                // Segunda confirmación
                case FIN:
                    // Proceso en estado pasivo
                    estado = PASIVO;

                    if (token_presente) {
                        if (rank == 0) {
                            color_token = BLANCO;
                        } else if (color == NEGRO) {
                            color_token = NEGRO;
                        }

                        color = BLANCO;
                        token_presente = false;
                        activo = false;
                        MPI_Send(color_token, 1, MPI_INT, anterior, FIN, MPI_COMM_WORLD);
                    }

                    break;
            }
        }
    }

    // El proceso tiene nodos para trabajar
    if (activo) {
        // Sondear si hay mensajes pendientes de otros procesos
        flag = 1;
        while (flag > 0) {
            MPI_Iprobe(MPI_ANY_SOURCE, PETICION, MPI_COMM_WORLD, &flag, &estado_mensaje);

            // Se ha recibido un mensaje
            if (flag > 0) {
                // Recibir mensaje de petición de trabajo
                MPI_Recv(&p_trabajo, 1, MPI_INT, estado_mensaje.MPI_SOURCE, PETICION, MPI_COMM_WORLD, &estado_mensaje);

                // Enviar la mitad de mis nodos
                if (pila.tamanio > UMBRAL_PILA) {
                    if (rank < estado_mensaje.MPI_SOURCE) color = NEGRO;

                    pila.divide(&pila2);
                    MPI_Send(pila2.nodos, pila2.tamanio, MPI_INT, estado_mensaje.MPI_SOURCE, NODOS, MPI_COMM_WORLD);
                } else {
                    // Pasar petición de trabajo al siguiente proceso
                    MPI_Send(&p_trabajo, 1, MPI_INT, siguiente, PETICION, MPI_COMM_WORLD);
                }
            }
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
    color = BLANCO;

    // Proceso 0 lee la matriz del problema inicial
    if (rank == 0) {
        // El proceso 0 empieza teniendo el token
        token_presente = true;
        color_token = BLANCO;
        estado = ACTIVO;

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
        estado = PASIVO;

        // Realizar peticiones de trabajo
        Equilibrar_Carga(pila, activo);

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

        Equilibrar_Carga(pila, activo);
        if (activo) {
            pila.pop(nodo);
        }

        iteraciones++;
    }

    t = MPI_Wtime() - t;

    // Obtener iteraciones totales
    // int total_iteraciones = 0;
    // MPI_Reduce();

    MPI_Finalize();

    if (rank == 0) {
        cout << "Solucion: \n" << cout;
        EscribeNodo(&solucion);
        cout << "Tiempo gastado = " << t << endl;
        cout << "Numero de iteraciones = " << iteraciones << endl << endl;
    }

    liberarMatriz(tsp0);
}
