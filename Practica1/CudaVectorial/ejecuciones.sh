#!/bin/bash
# Ejecución completa del ejercicio 2 de la práctica 1

# Compilar y eliminar resultados anteriores
make
rm resultados/tiempos.dat
rm resultados/grafica*

# Tamaño del problema
N=200000

# Ejecutar algoritmo de Floyd con diferentes tamaño de bloques CUDA 
for blocksize in 256 512 1024 
do
    nvidia_optimus ./vectorial $N $blocksize >> "resultados/tiempos.dat"
done

# Generar gráficas
gnuplot -p graficas.gp