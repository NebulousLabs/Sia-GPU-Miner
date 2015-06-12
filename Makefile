LFLAGS = -lOpenCL -lcurl
CFLAGS = -std=c11 -Wall -pedantic -O2
OBJS = sia-gpu-miner.o network.o
CXX = gcc

%.o: %.c network.h
	$(CXX) -c -s $(CFLAGS) $< -o $@

all: sia-gpu-miner

sia-gpu-miner: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LFLAGS)

clean:
	rm sia-gpu-miner $(OBJS)

.PHONY: all sia-gpu-miner clean
