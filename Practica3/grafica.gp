set title "Tiempos de ejecución de algoritmo B&B con distintos procesadores"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_tiempos.png"

# columna en eje x : columna en eje y
set xlabel "Tamaño del problema (N)"
set ylabel "Tiempo de ejecución"
plot "resultados/resultados.dat" using 1:2 title "tSec" with lp ps 4 lw 4, \
 "resultados/resultados.dat" using 1:3 title "tParalelo (P=2)" with lp ps 4 lw 4, \
  "resultados/resultados.dat" using 1:5 title "tParalelo (P=3)" with lp ps 4 lw 4

set title "Ganancias del algoritmo B&B con distintos procesadores con respecto a la ejecución secuencial"
set term pngcairo dashed size 1400,1050
set output "resultados/grafica_ganancias.png"

set xlabel "Tamaño del problema (N)"
set ylabel "Ganancia (S)"
plot "resultados/resultados.dat" using 1:4 title "Ganancia (P=2)" with lp ps 4 lw 4, \
 "resultados/resultados.dat" using 1:6 title "Ganancia (P=3)" with lp ps 4 lw 4