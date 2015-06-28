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

uint64_t *blockHeadermobj = nullptr;
uint64_t *headerHashmobj = nullptr;
uint64_t *nonceOutmobj = nullptr;
uint64_t *vpre = nullptr;
cudaError_t ret;
cudaStream_t cudastream;

unsigned int blocks_mined = 0;
static volatile int quit = 0;
bool target_corrupt_flag = false;

#define rotr64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))

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

double target_to_diff(const uint32_t *const target)
{
	// we are lazy and only take the most significant 64 bits
	return (4294967296.0 * 0xffff0000) / ((double)swap32(target[2]) + ((double)swap32(target[1]) * 4294967296.0));
}

// Perform global_item_size * iter_per_thread hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
double grindNonces(uint32_t items_per_iter, int cycles_per_iter)
{
	static bool init = false;
	static uint8_t *headerHash = nullptr;
	static uint32_t *target = nullptr;
	static uint64_t *nonceOut = nullptr;
	static uint8_t *blockHeader = nullptr;
	static uint64_t *v1 = nullptr;

	if(!init)
	{
		cudaMallocHost(&headerHash, 32);
		cudaMallocHost(&target, 32);
		cudaMallocHost(&nonceOut, 8);
		cudaMallocHost(&blockHeader, 80);
		cudaMallocHost(&v1, 16*8);
		ret = cudaGetLastError();
		if(ret != cudaSuccess)
		{
			printf("grindNonces init failed: %s\n", cudaGetErrorString(ret)); exit(1);
		}
		init = true;
	}

	// Start timing this iteration
	chrono::time_point<chrono::system_clock> startTime, endTime;
	startTime = chrono::system_clock::now();

	int i;

	// Get new block header and target
	if(get_header_for_work((uint8_t*)target, blockHeader) != 0)
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
		printf("Usually this resolves itself within a minute or so\n");
		printf("If it happens frequently trying increasing seconds per iteration\n");
		printf("e.g. \"./gpu-miner -s 3 -c 200\"\n");
		printf("Waiting for problem to be resolved...");
		fflush(stdout);
		return -1;
	}
	target_corrupt_flag = 0;
	*nonceOut = 0;

	v1[0] = 0x6A09E667F2BDC928u + 0x510e527fade682d1u + ((uint64_t*)blockHeader)[0]; v1[12] = rotr64(0x510E527FADE68281u ^ v1[0], 32); v1[8] = 0x6a09e667f3bcc908u + v1[12]; v1[4] = rotr64(0x510e527fade682d1u ^ v1[8], 24);
	v1[0] = v1[0] + v1[4] + ((uint64_t*)blockHeader)[1]; v1[12] = rotr64(v1[12] ^ v1[0], 16); v1[8] = v1[8] + v1[12]; v1[4] = rotr64(v1[4] ^ v1[8], 63);
	v1[1] = 0xbb67ae8584caa73bu + 0x9b05688c2b3e6c1fu + ((uint64_t*)blockHeader)[2]; v1[13] = rotr64(0x9b05688c2b3e6c1fu ^ v1[1], 32); v1[9] = 0xbb67ae8584caa73bu + v1[13]; v1[5] = rotr64(0x9b05688c2b3e6c1fu ^ v1[9], 24);
	v1[1] = v1[1] + v1[5] + ((uint64_t*)blockHeader)[3]; v1[13] = rotr64(v1[13] ^ v1[1], 16); v1[9] = v1[9] + v1[13]; v1[5] = rotr64(v1[5] ^ v1[9], 63);

	ret = cudaMemcpyAsync(vpre, v1, 16 * 8, cudaMemcpyHostToDevice, cudastream);
	if(ret != cudaSuccess)
	{
		printf("failed to write vpre buffer: %s\n", cudaGetErrorString(ret)); exit(1);
	}

	for(i = 0; i < cycles_per_iter; i++)
	{
		blockHeader[38] = i / 256;
		blockHeader[39] = i % 256;

		// Copy input data to the memory buffer
		ret = cudaMemcpyAsync(blockHeadermobj, blockHeader, 80, cudaMemcpyHostToDevice, cudastream);
		if(ret != cudaSuccess)
		{
			printf("failed to write to blockHeadermobj buffer: %s\n", cudaGetErrorString(ret)); exit(1);
		}
		ret = cudaMemcpyAsync(nonceOutmobj, nonceOut, 8, cudaMemcpyHostToDevice, cudastream);
		if(ret != cudaSuccess)
		{
			printf("failed to write nonce to buffer: %s\n", cudaGetErrorString(ret)); exit(1);
		}

		extern void nonceGrindcuda(cudaStream_t, uint32_t, uint64_t *, uint64_t *, uint64_t *, uint64_t *);
		nonceGrindcuda(cudastream, items_per_iter, blockHeadermobj, headerHashmobj, nonceOutmobj, vpre);
		ret = cudaGetLastError();
		if(ret != cudaSuccess)
		{
			printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
		}

		// Copy result to host
		ret = cudaMemcpyAsync(headerHash, headerHashmobj, 32, cudaMemcpyDeviceToHost, cudastream);
		if(ret != cudaSuccess)
		{
			printf("failed to read header hash from buffer: %s\n", cudaGetErrorString(ret)); exit(1);
		}

		ret = cudaMemcpyAsync(nonceOut, nonceOutmobj, 8, cudaMemcpyDeviceToHost, cudastream);
		if(ret != cudaSuccess)
		{
			printf("failed to read nonce from buffer: %s\n", cudaGetErrorString(ret)); exit(1);
		}
		cudaStreamSynchronize(cudastream);

		if(*nonceOut != 0)
		{
			int j = 4;
			while(headerHash[j] == ((uint8_t*)target)[j] && j<32)
				j++;
			if(j==32 || headerHash[j] < ((uint8_t*)target)[j])
			{
				// Copy nonce to header.
				((uint64_t*)blockHeader)[4] = *nonceOut;
				if(submit_header(blockHeader))
					blocks_mined++;
				return -1;
			}
			*nonceOut = 0;
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
	char *tmp;
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
			printf("\t default: %d\n", cycles_per_iter);
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
			cycles_per_iter = strtoul(optarg, &tmp, 10);
			if(cycles_per_iter < 1 || cycles_per_iter > 1000)
			{
				printf("Cycles must be at least 1 and no more than 1000\n");
				exit(1);
			}
			break;
		case 's':
			seconds_per_iter = strtod(optarg, &tmp);
			break;
		case 'p':
			port_number = _strdup(optarg);
			break;
		case 'd':
			deviceid = strtoul(optarg, &tmp, 10);
			break;
		}
	}

	// Set siad URL
	network_init(port_number);

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
		printf("Device %d: %s (Compute Capability %d.%d)\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	printf("\nUsing device %d\n", deviceid);

	ret = cudaSetDevice(deviceid);
	if(ret != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(ret)); exit(1);
	}
	cudaDeviceReset();
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
		printf("failed to create blockHeadermobj buffer: %s\n", cudaGetErrorString(ret)); exit(1);
	}
	ret = cudaMalloc(&headerHashmobj, 32);
	if(ret != cudaSuccess)
	{
		printf("failed to create headerHashmobj buffer: %s\n", cudaGetErrorString(ret)); exit(1);
	}
	ret = cudaMalloc(&nonceOutmobj, 8);
	if(ret != cudaSuccess)
	{
		printf("failed to create nonceOutmobj buffer: %s\n", cudaGetErrorString(ret)); exit(1);
	}
	ret = cudaMalloc(&vpre, 16*8);
	if(ret != cudaSuccess)
	{
		printf("failed to create vpre buffer: %s\n", cudaGetErrorString(ret)); exit(1);
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

	network_cleanup();

	return EXIT_SUCCESS;
}
