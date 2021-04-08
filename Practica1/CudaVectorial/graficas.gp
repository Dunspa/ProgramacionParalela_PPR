set title "Tiempos de ejecución de la transformación vectorial en CPU y GPU (shared y global) con respecto al Blocksize"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_tiempos.png"

# columna en eje x : columna en eje y
set xlabel "Blocksize"
set ylabel "Tiempo de ejecución"
plot "resultados/tiempos.dat" using 1:2 title "tcpu" with lp ps 2, \
 "resultados/tiempos.dat" using 1:3 title "tgpuSHARED" with lp ps 2, \
  "resultados/tiempos.dat" using 1:4 title "tgpuGLOBAL" with lp ps 2
      

set title "Ganancia en velocidad de las versiones en GPU con respecto a la versión monohebra en CPU con respecto al Blocksize"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_ganancia.png"

set xlabel "Blocksize"
set ylabel "Ganancia (S)"
plot "resultados/tiempos.dat" using 1:5 title "S_SHARED" with lp ps 2, \
 "resultados/tiempos.dat" using 1:6 title "S_GLOBAL" with lp ps 2