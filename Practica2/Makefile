# Makefile

all:matrizvector_1d matrizvector_2d

matrizvector_1d:matrizvector_1d.cc
	mpiCC matrizvector_1d.cc -o matrizvector_1d

matrizvector_2d:matrizvector_2d.cc
	mpiCC matrizvector_2d.cc -o matrizvector_2d

clean: 
	rm matrizvector_1d matrizvector_2d resultados/*