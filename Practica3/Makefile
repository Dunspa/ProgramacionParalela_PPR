# Makefile

all: bbseq bbpar

bbpar: bbpar.o libbb.o
	mpicxx -O3 bbpar.o libbb.o -o bbpar

bbpar.o: bbpar.cc
	mpicxx -O3 -c bbpar.cc

bbseq: bbseq.o libbb.o
	mpicxx -O3 bbseq.o libbb.o -o bbseq
	
bbseq.o: bbseq.cc
	mpicxx -O3 -c bbseq.cc

libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc 

clean:
	rm -f *.o bbseq bbpar