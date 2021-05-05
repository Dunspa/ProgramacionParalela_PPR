#!/bin/bash
# Ejecución completa de la práctica 2

# Compilar y eliminar resultados anteriores
make
rm resultados/tiempos*
rm resultados/grafica*

# Ejecutar el algoritmo 1d para distinto tamaño del problema
for n in 300 600 900 1200 1400
do
    echo -n $n "" >> "resultados/tiempos_1d.dat"
    mpirun --oversubscribe -np 4 matrizvector_1d $n 1 >> "resultados/tiempos_1d.dat"
    mpirun --oversubscribe -np 9 matrizvector_1d $n 1 >> "resultados/tiempos_1d.dat"
    echo "" >> "resultados/tiempos_1d.dat"
done

# Ejecutar el algoritmo 2d para distinto tamaño del problema
for n in 300 600 900 1200 
do
    echo -n $n "" >> "resultados/tiempos_2d.dat"
    mpirun --oversubscribe -np 4 matrizvector_2d $n 1 >> "resultados/tiempos_2d.dat"
    mpirun --oversubscribe -np 9 matrizvector_2d $n 1 >> "resultados/tiempos_2d.dat"
    echo "" >> "resultados/tiempos_2d.dat"
done

echo -n 1400 "" >> "resultados/tiempos_2d.dat"
mpirun --oversubscribe -np 4 matrizvector_2d 1400 1 >> "resultados/tiempos_2d.dat"

# Generar gráficas
gnuplot -p grafica.gp