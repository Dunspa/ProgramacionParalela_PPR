set title "Tiempos de ejecución del producto matriz-vector con descomposición unidimensional"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_1d.png"

# columna en eje x : columna en eje y
set xlabel "Tamaño del problema (N)"
set ylabel "Tiempo de ejecución"
plot "resultados/tiempos_1d.dat" using 1:2 title "tSec" with lp ps 4 lw 4, \
 "resultados/tiempos_1d.dat" using 1:3 title "tParalelo (P=4)" with lp ps 4 lw 4, \
  "resultados/tiempos_1d.dat" using 1:6 title "tParalelo (P=9)" with lp ps 4 lw 4

set title "Ganancias del proucto matriz-vector con descomposición unidimensional"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_ganancias_1d.png"

set xlabel "Tamaño del problema (N)"
set ylabel "Ganancia (S)"
plot "resultados/tiempos_1d.dat" using 1:4 title "Ganancia (P=4)" with lp ps 4 lw 4, \
 "resultados/tiempos_1d.dat" using 1:7 title "Ganancia (P=9)" with lp ps 4 lw 4

set title "Tiempos de ejecución del producto matriz-vector con descomposición bidimensional"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_2d.png"

# columna en eje x : columna en eje y
set xlabel "Tamaño del problema (N)"
set ylabel "Tiempo de ejecución"
plot "resultados/tiempos_2d.dat" using 1:2 title "tSec" with lp ps 4 lw 4, \
 "resultados/tiempos_2d.dat" using 1:3 title "tParalelo (P=4)" with lp ps 4 lw 4, \
  "resultados/tiempos_2d.dat" using 1:6 title "tParalelo (P=9)" with lp ps 4 lw 4

set title "Ganancias del proucto matriz-vector con descomposición bidimensional"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_ganancias_2d.png"

set xlabel "Tamaño del problema (N)"
set ylabel "Ganancia (S)"
plot "resultados/tiempos_2d.dat" using 1:4 title "Ganancia (P=4)" with lp ps 4 lw 4, \
 "resultados/tiempos_2d.dat" using 1:7 title "Ganancia (P=9)" with lp ps 4 lw 4