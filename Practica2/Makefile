# Makefile

all:matrizvector_secuencial matrizvector_1d

matrizvector_secuencial:matrizvector_secuencial.cc
	g++ matrizvector_secuencial.cc -o matrizvector_secuencial

matrizvector_1d:matrizvector_1d.cc
	mpiCC matrizvector_1d.cc -o matrizvector_1d

clean: 
	rm matrizvector_secuencial matrizvector_1d