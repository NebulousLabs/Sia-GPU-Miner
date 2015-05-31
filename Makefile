LFLAGS = -lOpenCL -lcurl
CFLAGS = -std=c11 -Wall -pedantic
OBJS = gpu-miner.o network.o
CXX = gcc

%.o: %.c network.h
	$(CXX) -c -s $(CFLAGS) $< -o $@

all: gpu-miner

gpu-miner: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LFLAGS)

clean:
	rm gpu-miner $(OBJS)

.PHONY: all gpu-miner clean
