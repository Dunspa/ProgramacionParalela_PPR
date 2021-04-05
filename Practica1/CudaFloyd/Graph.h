#ifndef GRAPH_H
#define GRAPH_H

const int INF = 1000000;

// Clase Grafo, con lista de adyacencias
class Graph {
private:
	int *A;

public:
	int vertices;

    Graph();
	void fija_nverts(const int verts);
	void inserta_arista(const int ptA,const int ptB, const int edge);
	int arista(const int ptA,const int ptB);
    void imprime();
    void lee(char *filename);
	int * Get_Matrix(){return A;}
};


#endif