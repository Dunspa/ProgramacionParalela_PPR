set title "Tiempos de ejecución del algoritmo de Floyd en CPU y GPU (1d y 2d) para distintos BSize"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_tiempos.png"

# columna en eje x : columna en eje y
set xlabel "Tamaño del problema (N)"
set ylabel "Tiempo de ejecución"
plot "resultados/tiempos_8.dat" using 1:2 title "tcpu(BSIZE=64)" with lp ps 2, \
 "resultados/tiempos_8.dat" using 1:3 title "tgpu1d(BSIZE=64)" with lp ps 2, \
  "resultados/tiempos_8.dat" using 1:4 title "tgpu2d(BSIZE=64)" with lp ps 2, \
   "resultados/tiempos_16.dat" using 1:2 title "tcpu(BSIZE=256)" with lp ps 2, \
    "resultados/tiempos_16.dat" using 1:3 title "tgpu1d(BSIZE=256)" with lp ps 2, \
     "resultados/tiempos_16.dat" using 1:4 title "tgpu2d(BSIZE=256)" with lp ps 2, \
      "resultados/tiempos_32.dat" using 1:2 title "tcpu(BSIZE=1024)" with lp ps 2, \
       "resultados/tiempos_32.dat" using 1:3 title "tgpu1d(BSIZE=1024)" with lp ps 2, \
        "resultados/tiempos_32.dat" using 1:4 title "tgpu2d(BSIZE=1024)" with lp ps 2

set title "Ganancia en velocidad de las versiones en GPU con respecto a la versión monohebra en CPU para distintos BSize"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_ganancia.png"

set xlabel "Tamaño del problema (N)"
set ylabel "Ganancia (S)"
plot "resultados/tiempos_8.dat" using 1:5 title "S1d(BSIZE=64)" with lp ps 2, \
 "resultados/tiempos_8.dat" using 1:6 title "S2d(BSIZE=64)" with lp ps 2, \
  "resultados/tiempos_16.dat" using 1:5 title "S1d(BSIZE=256)" with lp ps 2, \
   "resultados/tiempos_16.dat" using 1:6 title "S2d(BSIZE=256)" with lp ps 2, \
    "resultados/tiempos_32.dat" using 1:5 title "S1d(BSIZE=1024)" with lp ps 2, \
     "resultados/tiempos_32.dat" using 1:6 title "S2d(BSIZE=1024)" with lp ps 2