all: main

# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

FFLAGS = 

CPPFLAGS = -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -ldl -lrt -O3
CPPFLAGS += -std=c99 -g -Wall -lm

FPPFLAGS = 

SOURCECPP = main.c
OBJSC = main.o

override CC=mpicc

main: $(OBJSC)
	$(CC) -o $@ $^ $(CPPFLAGS)

.PHONY: clean
clean:
	rm -f  $(OBJSC) main
