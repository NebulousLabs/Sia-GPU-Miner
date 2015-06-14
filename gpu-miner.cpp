#ifdef __linux__
#define _GNU_SOURCE
#define _POSIX_SOURCE
#include <sys/time.h>
#endif

#ifdef _MSC_VER
extern "C" {
	int getopt(int, char * const *, const char *);
	extern char *optarg;
}
#endif

#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <chrono>
using namespace std;
#include <signal.h>
#ifdef _MSC_VER
#include "VisualStudio/getopt/getopt.h"
#else
#include <getopt.h>
#endif
//#include <unistd.h>
#include <cuda_runtime.h>

#include "network.h"

#define MAX_SOURCE_SIZE (0x200000)

char *blockHeadermobj = nullptr;
char *headerHashmobj = nullptr;
char *targmobj = nullptr;
char *nonceOutmobj = nullptr;
cudaError_t ret;
cudaStream_t cudastream;

CURL *curl = nullptr;

unsigned int blocks_mined = 0;
static volatile int quit = 0;
bool target_corrupt_flag = false;

void quitSignal(int __unused)
{
	quit = 1;
	printf("\nCaught deadly signal, quitting...\n");
}

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define LINUX_BSWAP
#endif

static inline uint32_t swap32(uint32_t x)
{
#ifdef LINUX_BSWAP
	return __builtin_bswap32(x);
#else
#ifdef _MSC_VER
	return _byteswap_ulong(x);
#else
	return ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu));
#endif
#endif
}

// Perform global_item_size * iter_per_thread hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
double grindNonces(uint32_t items_per_iter, int cycles_per_iter)
{
	static bool init = false;
	static uint8_t *headerHash = nullptr;
	static uint32_t *target = nullptr;
	static uint8_t *nonceOut = nullptr;
	static uint8_t *blockHeader = nullptr;

	if(!init)
	{
		cudaMallocHost(&headerHash, 32);
		cudaMallocHost(&target, 32);
		cudaMallocHost(&nonceOut, 8);
		cudaMallocHost(&blockHeader, 80);
		ret = cudaGetLastError();
		if(ret != cudaSuccess)
		{
			printf("%s\n", cudaGetErrorString(ret)); exit(1);
		}
		init = true;
	}

	// Start timing this iteration
	chrono::time_point<chrono::system_clock> startTime, endTime;
	startTime = chrono::system_clock::now();

	int i;

	// Get new block header and target
	if(get_header_for_work(curl, (uint8_t*)target, blockHeader) != 0)
	{
		return 0;
	}

	// Check for target corruption
	if(target[0] != 0)
	{
		if(target_corrupt_flag)
		{
			return -1;
		}
		target_corrupt_flag = true;
		printf("\nReceived corrupt target from Sia\n");
		printf("%08x %08x %08x %08x %08x %08x %08x %08x \n", swap32(target[0]), swap32(target[1]), swap32(target[2]), swap32(target[3]), swap32(target[4]), swap32(target[5]), swap32(target[6]), swap32(target[7]));
		printf("Usually this resolves itself within a minute or so\n");
		printf("If it happens frequently trying increasing seconds per iteration\n");
		printf("e.g. \"./gpu-miner -s 3 -c 200\"\n");
		printf("Waiting for problem to be resolved...");
		fflush(stdout);
	}
	target_corrupt_flag = 0;
	*((uint64_t*)nonceOut) = 0;

	for(i = 0; i < cycles_per_iter; i++)
	{
		blockHeader[38] = i / 256;
		blockHeader[39] = i % 256;

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
		ret = cudaMemcpyAsync(nonceOutmobj, nonceOut, 8, cudaMemcpyHostToDevice, cudastream);
		if(ret != cudaSuccess)
		{
			printf("failed to read nonce from buffer: %d\n", ret); exit(1);
		}

		extern void nonceGrindcuda(cudaStream_t, int, char *, char *, char *, char *);
		nonceGrindcuda(cudastream, items_per_iter, blockHeadermobj, headerHashmobj, targmobj, nonceOutmobj);
		ret = cudaGetLastError();
		if(ret != cudaSuccess)
		{
			printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
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
		cudaStreamSynchronize(cudastream);

		if(*((uint64_t*)nonceOut) != 0)
		{
			i = 4;
			while(headerHash[i] <= ((uint8_t*)target)[i] && i<32)
				i++;
			if(i == 32)
			{

				// Copy nonce to header.
				((uint64_t*)blockHeader)[4] = *((uint64_t*)nonceOut);
				submit_header(curl, blockHeader);
				blocks_mined++;
				return -1;
			}
			*((uint64_t*)nonceOut) = 0;
		}
	}

	// Hashrate is inaccurate if a block was found
	endTime=chrono::system_clock::now();
	double elapsedTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() / 1000000.0;
	double hash_rate = cycles_per_iter * (double)items_per_iter / elapsedTime / 1000000;

	return hash_rate;
}

int main(int argc, char *argv[])
{
	int c;
	unsigned int deviceid = 0;
	cudaDeviceProp deviceProp;
	char *port_number = "9980";
	double hash_rate;
	uint32_t items_per_iter = 256 * 256 * 256 * 16;

	// parse args
	unsigned int cycles_per_iter = 15;
	double seconds_per_iter = 10.0;
	while((c = getopt(argc, argv, "hc:s:p:d:")) != -1)
	{
		switch(c)
		{
		case 'h':
			printf("\nUsage:\n\n");
			printf("\t c - cycles: number of hashing loops between API calls\n");
			printf("\t default: %f\n", cycles_per_iter);
			printf("\t\tIncrease this if your computer is freezing or locking up\n");
			printf("\n");
			printf("\t s - seconds between Sia API calls and hash rate updates\n");
			printf("\t default: %f\n", seconds_per_iter);
			printf("\n");
			printf("\t d - device: the device id of the card you want to use");
			printf("\n");
			exit(0);
			break;
		case 'c':
			sscanf(optarg, "%d", &cycles_per_iter);
			if(cycles_per_iter < 1 || cycles_per_iter > 1000)
			{
				printf("Cycles must be at least 1 and no more than 1000\n");
				exit(1);
			}
			break;
		case 's':
			sscanf(optarg, "%lf", &seconds_per_iter);
			break;
		case 'p':
			port_number = _strdup(optarg);
			break;
		case 'd':
			sscanf(optarg, "%u", &deviceid);
			break;
		}
	}

	// Set siad URL
	set_port(port_number);

	// Use curl to communicate with siad
	curl = curl_easy_init();
	printf("\nInitializing...\n");

	int deviceCount;
	ret = cudaGetDeviceCount(&deviceCount);
	if(ret != cudaSuccess)
	{
		if(ret == cudaErrorNoDevice)
			printf("No CUDA device found");
		if(ret == cudaErrorInsufficientDriver)
			printf("Driver error\n");
		return -1;
	}
	for(int device = 0; device<deviceCount; ++device)
	{
		ret = cudaGetDeviceProperties(&deviceProp, device);
		if(ret != cudaSuccess)
		{
			printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
		}
		printf("Device %d: %s (Compute Capability %d.%d)", device, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	printf("\nUsing device %d\n", deviceid);

	ret = cudaSetDevice(deviceid);
	if(ret != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
	}

	ret = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	if(ret != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
	}

	ret = cudaStreamCreate(&cudastream);
	if(ret != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
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

	chrono::time_point<chrono::system_clock> startTime, endTime;
	startTime = chrono::system_clock::now();

	grindNonces(items_per_iter, 1);

	endTime=chrono::system_clock::now();
	double elapsedTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() / 1000000.0;
	items_per_iter *= (seconds_per_iter / elapsedTime) / cycles_per_iter;

	// Grind nonces until SIGINT
	signal(SIGINT, quitSignal);
	while(!quit)
	{
		// Repeat until no block is found
		do
		{
			hash_rate = grindNonces(items_per_iter, cycles_per_iter);
		} while(hash_rate == -1);

		if(!quit && hash_rate != 0)
		{
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
	}

	// Finalization
	ret = cudaStreamDestroy(cudastream);
	if(ret != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
	}
	cudaDeviceReset();

	curl_easy_cleanup(curl);

	return EXIT_SUCCESS;
}
