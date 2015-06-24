all-flags = -c -std=c11 -Wall -pedantic -O2
gcc-libs = -lOpenCL -lcurl
clang-libs =  -lcurl -framework OpenCL

all:
	@echo "commands: clean, linux, mac"

linux:
	gcc $(all-flags) sia-gpu-miner.c -o sia-gpu-miner.o
	gcc $(all-flags) network.c -o network.o
	gcc sia-gpu-miner.o network.o -o sia-gpu-miner $(gcc-libs)

mac:
	clang $(all-flags) sia-gpu-miner.c -o sia-gpu-miner.o
	clang $(all-flags) network.c -o network.o
	clang sia-gpu-miner.o network.o -o sia-gpu-miner $(clang-libs)

clean:
	rm sia-gpu-miner *.o

.PHONY: all linux mac clean
