#!/bin/bash
# Ejecución completa del ejercicio 1 de la práctica 1

# Compilar y eliminar resultados anteriores
make
rm resultados/tiempos*

# Ejecutar algoritmo de Floyd con diferentes tamaño de bloques CUDA 
# (8x8, 16x16, 32x32)
for blocksize in 8 16 32 
do
    # Diferente tamaño de problema
    for input in input400 input1000 input1400 input2000
    do
        nvidia_optimus ./floyd input/$input $blocksize >> "resultados/tiempos_${blocksize}.dat"
    done
done

# Generar gráficas
# gnuplot -p graficas.gp