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

// Tipos de mensajes que se envían los procesos
const int PETICION = 0;
const int NODOS = 1;
const int TOKEN = 2;
const int FIN = 3;

// Estados en los que se puede encontrar un proceso
const int ACTIVO = 0;
const int PASIVO = 1;

// Colores que pueden tener tanto los procesos como el token
const int BLANCO = 0;
const int NEGRO = 1;

// No ceder nodos si el tamaño de la pila está por debajo del umbral
const int UMBRAL_PILA = 2;

int estado;           // Estado del proceso {ACTIVO, PASIVO}
int color;            // Color del proceso {BLANCO,NEGRO}
int color_token;      // Color del token la última vez que estaba en poder del procesos
bool token_presente;  // Indica si el proceso posee el token
int anterior;         // Identificador del anterior proceso
int siguiente;        // Identificador del siguiente proceso

unsigned int NCIUDADES;
int tag, flag;
int p_trabajo;  // Proceso al que enviar el trabajo
MPI_Status estado_mensaje;

// Algoritmo de equilibrado de carga con detección de fin, con paso por referencia
void Equilibrar_Carga(tPila& pila, bool& activo, tNodo& solucion, int rank) {
    // El proceso no tiene nodos para trabajar: pedir nodos a otros procesos
    if (pila.vacia()) {
        // Enviar mensaje de petición de trabajo
        MPI_Send(&rank, 1, MPI_INT, siguiente, PETICION, MPI_COMM_WORLD);

        // Esperar a conseguir trabajo
        while (pila.vacia() && activo) {
            // Sondear mensajes
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado_mensaje);

            // Tipos de mensajes que sse envían los procesos
            switch (estado_mensaje.MPI_TAG) {
                // Petición de trabajo
                case PETICION:
                    // Recibir mensaje de petición de trabajo
                    MPI_Recv(&p_trabajo, 1, MPI_INT, estado_mensaje.MPI_SOURCE, PETICION, MPI_COMM_WORLD, &estado_mensaje);

                    if (p_trabajo == rank) {  // Petición devuelta
                        // Proceso en estado pasivo
                        estado = PASIVO;

                        // Reiniciar detección de fin
                        if (token_presente) {
                            if (rank == 0) {  // Si es el proceso 0, se prueba a empezar a detectar el fin
                                color_token = BLANCO;
                            } else if (color == NEGRO) {
                                color_token = NEGRO;
                            }

                            token_presente = false;
                            color = BLANCO;

                            // Se envía el token al proceso anterior
                            MPI_Send(&color_token, 1, MPI_INT, anterior, TOKEN, MPI_COMM_WORLD);
                        }
                    }

                    // Pasar petición de trabajo al siguiente proceso
                    MPI_Send(&p_trabajo, 1, MPI_INT, siguiente, PETICION, MPI_COMM_WORLD);

                    break;

                // Resultado de una petición de trabajo
                case NODOS:
                    // Proceso en estado activo
                    estado = ACTIVO;

                    // Recibir nodos del proceso donante
                    int numnodos;
                    MPI_Get_count(&estado_mensaje, MPI_INT, &numnodos); // Obtener dinámicamente el tamaño del mensaje
                    MPI_Recv(pila.nodos, numnodos, MPI_INT, estado_mensaje.MPI_SOURCE, NODOS, MPI_COMM_WORLD,
                             &estado_mensaje);
                    pila.tope = numnodos;

                    break;

                // Se recibe el token
                case TOKEN:
                    // El proceso tiene el token
                    token_presente = true;
                    MPI_Recv(&color_token, 1, MPI_INT, estado_mensaje.MPI_SOURCE, TOKEN, MPI_COMM_WORLD, &estado_mensaje);

                    // Si el proceso es pasivo, se envía el token al proceso anterior, si no, lo mantiene
                    if (estado == PASIVO) {
                        if (rank == 0 && color == BLANCO && color_token == BLANCO) {  // Terminación detectada
                            // Recibir mensajes de petición pendientes
                            flag = 1;
                            while (flag > 0) {
                                MPI_Iprobe(MPI_ANY_SOURCE, PETICION, MPI_COMM_WORLD, &flag, &estado_mensaje);

                                if (flag > 0) {
                                    MPI_Recv(&p_trabajo, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, MPI_COMM_WORLD,
                                             &estado_mensaje);
                                }
                            }

                            // Envío de la solución para comenzar la segunda confirmación de fin
                            MPI_Send(solucion.datos, 2 * NCIUDADES, MPI_INT, siguiente, FIN, MPI_COMM_WORLD);

                            // Recibir último mensaje fin
                            tNodo nueva_solucion;
                            MPI_Recv(nueva_solucion.datos, 2 * NCIUDADES, MPI_INT, anterior, FIN,
                                     MPI_COMM_WORLD, &estado_mensaje);

                            // Quedarme con la mejor solución de las últimas generadas por cada proceso
                            if (nueva_solucion.ci() < solucion.ci() && nueva_solucion.ci() > 0) {
                                CopiaNodo(&nueva_solucion, &solucion);
                            }

                            // Terminar ejecución
                            activo = false;
                        } else {
                            // Cambiar el color al token
                            if (rank == 0) {
                                color_token = BLANCO;
                            } else if (color == NEGRO) {
                                color_token = NEGRO;
                            }

                            token_presente = false;
                            color = BLANCO;

                            // Se envía el token al proceso anterior
                            MPI_Send(&color_token, 1, MPI_INT, anterior, TOKEN, MPI_COMM_WORLD);
                        }
                    }

                    break;

                // Segunda confirmación
                case FIN:
                    // Proceso en estado pasivo
                    estado = PASIVO;

                    tNodo nueva_solucion;  // Nueva solución obtenida de otro proceso
                    MPI_Recv(nueva_solucion.datos, 2 * NCIUDADES, MPI_INT, estado_mensaje.MPI_SOURCE, FIN,
                             MPI_COMM_WORLD, &estado_mensaje);

                    // Quedarme con la mejor solución de las últimas generadas por cada proceso
                    if (nueva_solucion.ci() < solucion.ci() && nueva_solucion.ci() > 0) {
                        CopiaNodo(&nueva_solucion, &solucion);
                    }

                    // Enviar solución al siguiente
                    MPI_Send(solucion.datos, 2 * NCIUDADES, MPI_INT, siguiente, FIN, MPI_COMM_WORLD);

                    // Terminar ejecución
                    activo = false;

                    break;
            }
        }
    }

    // El proceso tiene nodos para trabajar
    if (activo) {
        // Sondear si hay mensajes pendientes de otros procesos
        flag = 1;
        while (flag > 0) {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &estado_mensaje);

            // Se ha recibido un mensaje
            if (flag > 0) {
                if (estado_mensaje.MPI_TAG == PETICION) {  // Responder a peticiones de trabajo enviando trabajo
                    // Recibir mensaje de petición de trabajo
                    MPI_Recv(&p_trabajo, 1, MPI_INT, estado_mensaje.MPI_SOURCE, PETICION, MPI_COMM_WORLD,
                             &estado_mensaje);

                    tPila pila2;  // Pila de nodos (trabajo) a enviar

                    // Enviar la mitad de mis nodos
                    if (pila.tamanio() > UMBRAL_PILA && pila.divide(pila2)) {
                        if (rank < p_trabajo) color = NEGRO;

                        MPI_Send(pila2.nodos, pila2.tope, MPI_INT, p_trabajo, NODOS, MPI_COMM_WORLD);
                    } else {
                        // Pasar petición de trabajo al siguiente proceso
                        MPI_Send(&p_trabajo, 1, MPI_INT, siguiente, PETICION, MPI_COMM_WORLD);
                    }
                } else if (estado_mensaje.MPI_TAG == TOKEN) {  // Guardar token para reenviarlo cuando sea pasivo
                    token_presente = true;

                    MPI_Recv(&color_token, 1, MPI_INT, estado_mensaje.MPI_SOURCE, TOKEN, MPI_COMM_WORLD,
                             &estado_mensaje);
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

    int rank, size;
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
    color_token = BLANCO;
    estado = ACTIVO;

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
        Equilibrar_Carga(pila, activo, solucion, rank);

        // No se cumple la condición de fin
        if (activo) {
            pila.pop(nodo);
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

        Equilibrar_Carga(pila, activo, solucion, rank);
        if (activo) {
            pila.pop(nodo);
        }

        iteraciones++;
    }

    t = MPI_Wtime() - t;

    MPI_Finalize();

    if (rank == 0) {
        cout << "Solucion: \n" << endl;
        EscribeNodo(&solucion);
        cout << "Tiempo gastado = " << t << endl;
    }
    cout << "Numero de iteraciones (proceso " << rank << ")= " << iteraciones << endl << endl;

    liberarMatriz(tsp0);
}
