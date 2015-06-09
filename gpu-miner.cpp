#ifdef __linux__
#define _GNU_SOURCE
#define _POSIX_SOURCE
#include <sys/time.h>
#endif

#include <ctime>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <iostream>
using namespace std;
#include <cuda_runtime.h>

#include "network.h"

#define MAX_SOURCE_SIZE (0x200000)

char *blockHeadermobj = NULL;
char *headerHashmobj = NULL;
char *targmobj = NULL;
char *nonceOutmobj = NULL;
cudaError_t ret;
cudaStream_t cudastream;

CURL *curl;

unsigned int blocks_mined = 0;


// Perform global_item_size * iter_per_thread hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
double grindNonces(size_t global_item_size)
{
	// Start timing this iteration
#ifdef __linux__
	struct timespec begin, end;
	clock_gettime(CLOCK_REALTIME, &begin);
#else
	clock_t startTime = clock();
#endif

	uint8_t blockHeader[80];
	uint8_t headerHash[32];
	uint8_t target[32];
	uint8_t nonceOut[8]; // This is where the nonce that gets a low enough hash will be stored

	int i;
	for(i = 0; i < 8; i++)
	{
		nonceOut[i] = 0;
		headerHash[i] = 255;
	}

	// Store block from siad
	uint8_t *block;
	size_t blocklen = 0;

	// Get new block header and target
	get_block_for_work(curl, target, blockHeader, &block, &blocklen);

	// Copy input data to the memory buffer
	ret = cudaMemcpyAsync(blockHeadermobj, blockHeader, 80, cudaMemcpyHostToDevice, cudastream);
	if(ret != cudaSuccess)
	{
		printf("failed to write to blockHeadermobj buffer: %d\n", ret); exit(1);
	}
	ret = cudaMemcpyAsync(headerHashmobj, headerHash, 32, cudaMemcpyHostToDevice, cudastream);
	if(ret != cudaSuccess)
	{
		printf("failed to write to headerHashmobj buffer: %d\n", ret); exit(1);
	}
	ret = cudaMemcpyAsync(targmobj, target, 32, cudaMemcpyHostToDevice, cudastream);
	if(ret != cudaSuccess)
	{
		printf("failed to write to targmobj buffer: %d\n", ret); exit(1);
	}

	// Execute OpenCL kernel as data parallel
	size_t local_item_size = 256;
	global_item_size -= global_item_size % 256;

	extern void nonceGrindcuda(cudaStream_t, int, int, char *, char *, char *, char *);
	nonceGrindcuda(cudastream, global_item_size, local_item_size, blockHeadermobj, headerHashmobj, targmobj, nonceOutmobj);
	ret = cudaGetLastError();
	if(ret != cudaSuccess)
	{
		cout << cudaGetErrorString(ret) << endl; return -1;
	}

	// Copy result to host
	ret = cudaMemcpyAsync(headerHash, headerHashmobj, 32, cudaMemcpyDeviceToHost, cudastream);
	if(ret != cudaSuccess)
	{
		printf("failed to read header hash from buffer: %d\n", ret); exit(1);
	}
	ret = cudaMemcpyAsync(nonceOut, nonceOutmobj, 8, cudaMemcpyDeviceToHost, cudastream);
	if(ret != cudaSuccess)
	{
		printf("failed to read nonce from buffer: %d\n", ret); exit(1);
	}
	cudaDeviceSynchronize();
	// Did we find one?
	i = 0;
	while(target[i] == headerHash[i] && i<32)
	{
		i++;
	}
	if(headerHash[i] < target[i] || i==32)
	{
		// Copy nonce to block
		for(i = 0; i < 8; i++)
		{
			block[i + 32] = nonceOut[i];
		}

		submit_block(curl, block, blocklen);
		blocks_mined++;
	}
	else
	{
		// Hashrate is inaccurate if a block was found
#ifdef __linux__
		clock_gettime(CLOCK_REALTIME, &end);

		double nanosecondsElapsed = 1e9 * (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec);
		double run_time_seconds = nanosecondsElapsed * 1e-9;
#else
		double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
#endif
		double hash_rate = global_item_size / (run_time_seconds * 1000000);
		// TODO: Print est time until next block (target difficulty / hashrate
		return hash_rate;
	}
	return -1;
}

int main()
{
	int i;
	size_t global_item_size = 1;

	// Use curl to communicate with siad
	curl = curl_easy_init();

	int deviceCount;
	ret = cudaGetDeviceCount(&deviceCount);
	if(ret != cudaSuccess)
	{
		if(ret == cudaErrorNoDevice)
			cout << "No CUDA device found" << endl;
		if(ret == cudaErrorInsufficientDriver)
			cout << "Driver error" << endl;
		return -1;
	}

	ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	if(ret != cudaSuccess)
	{
		cout << cudaGetErrorString(ret) << endl; return -1;
	}

	// make it the active device
	ret = cudaSetDevice(0);
	if(ret != cudaSuccess)
	{
		cout << cudaGetErrorString(ret) << endl; return -1;
	}

	ret = cudaStreamCreate(&cudastream);
		if(ret != cudaSuccess)
		{
			cout << cudaGetErrorString(ret) << endl; return -1;
		}
	// Create Buffer Objects
	ret = cudaMalloc(&blockHeadermobj, 80);
	if(ret != cudaSuccess)
	{
		printf("failed to create blockHeadermobj buffer: %d\n", ret); exit(1);
	}
	ret = cudaMalloc(&headerHashmobj, 32);
	if(ret != cudaSuccess)
	{
		printf("failed to create headerHashmobj buffer: %d\n", ret); exit(1);
	}
	ret = cudaMalloc(&targmobj, 32);
	if(ret != cudaSuccess)
	{
		printf("failed to create targmobj buffer: %d\n", ret); exit(1);
	}
	ret = cudaMalloc(&nonceOutmobj, 8);
	if(ret != cudaSuccess)
	{
		printf("failed to create nonceOutmobj buffer: %d\n", ret); exit(1);
	}

	double hash_rate;
	global_item_size = 256 * 256*8;

	// Make each iteration take about 3 seconds
#ifdef __linux__
	struct timespec begin, end;
	clock_gettime(CLOCK_REALTIME, &begin);
#else
	clock_t startTime = clock();
#endif
	grindNonces(global_item_size);
#ifdef __linux__
	clock_gettime(CLOCK_REALTIME, &end);

	double nanosecondsElapsed = 1e9 * (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec);
	double run_time_seconds = nanosecondsElapsed * 1e-9;
#else
	double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
#endif
	global_item_size *= 0.015 / run_time_seconds;

	// Grind nonces endlessly using
	while(1)
	{
		i++;
		double temp = grindNonces(global_item_size);
		while(temp == -1)
		{
			// Repeat until no block is found
			temp = grindNonces(global_item_size);
		}
		hash_rate = temp;
		if(i % 15 == 0)
		{
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
	}

	// Finalization
	ret = cudaStreamDestroy(cudastream);
	if(ret != cudaSuccess)
	{
		cout << cudaGetErrorString(ret) << endl; return -1;
	}
	cudaDeviceReset();

	curl_easy_cleanup(curl);

	return EXIT_SUCCESS;
}
