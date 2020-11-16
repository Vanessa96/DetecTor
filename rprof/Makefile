NVCC	:=nvcc --cudart=static -ccbin g++
CFLAGS	:=-O3 -std=c++14
INC_DIR	:= .
LIB_DIR	:=
LIBS	:=-lnvidia-ml

SOURCES := rprof \

all: $(SOURCES)
.PHONY: all

rprof: rprof.c
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
clean:
	rm -f $(SOURCES)
