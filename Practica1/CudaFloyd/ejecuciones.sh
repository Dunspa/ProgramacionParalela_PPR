#!/bin/bash

make
rm resultados/tiempos*

# Diferente tamaño de bloques CUDA (8x8, 16x16, 32x32)
for blocksize in 64 256 1024 
do
    # Diferente tamaño de problema
    for input in input400 input1000 input1400 input2000
    do
        nvidia_optimus ./floyd input/$input $blocksize >> "resultados/tiempos_${blocksize}.dat"
    done
done