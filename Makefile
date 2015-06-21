gcc-flags = -c -s -std=c11 -Wall -pedantic -O2
gcc-libs = -lOpenCL -lcurl
miner-version = v1.0.2

all: sia-gpu-miner

sia-gpu-miner:
	gcc $(gcc-flags) sia-gpu-miner.c -o sia-gpu-miner.o
	gcc $(gcc-flags) network.c -o network.o
	gcc -v sia-gpu-miner.o network.o -o sia-gpu-miner $(gcc-libs)

clean:
	rm sia-gpu-miner *.o

.PHONY: all sia-gpu-miner win clean
