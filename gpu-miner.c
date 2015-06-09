#ifdef __linux__
#define _GNU_SOURCE
#define _POSIX_SOURCE
#include <sys/time.h>
#endif

#include <time.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#include "network.h"
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x200000)

cl_command_queue command_queue = NULL;
cl_mem blockHeadermobj = NULL;
cl_mem headerHashmobj = NULL;
cl_mem targmobj = NULL;
cl_mem nonceOutmobj = NULL;
cl_kernel kernel = NULL;
cl_int ret;

CURL *curl;

unsigned int blocks_mined = 0;
static volatile int quit = 0;
int target_corrupt_flag = 0;

void quitSignal(int __unused) {
	quit = 1;
	printf("\nCaught deadly signal, quitting...\n");
}

// Perform items_per_iter * cycles_per_iter hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
double grindNonces(size_t items_per_iter, int cycles_per_iter) {
	// Start timing this iteration
	#ifdef __linux__
	struct timespec begin, end;
	clock_gettime(CLOCK_REALTIME, &begin);
	#else
	clock_t startTime = clock();
	#endif

	int i;
	uint8_t blockHeader[80];
	uint8_t headerHash[32];
	uint8_t target[32];
	uint8_t nonceOut[8]; // This is where the nonce that gets a low enough hash will be stored

	memset(nonceOut, 0, 8);
	memset(headerHash, 255, 32);
	memset(target, 255, 32);

	// Get new block header and target
	if (get_header_for_work(curl, target, blockHeader) != 0) {
		return 0;
	}

	// Check for target corruption
	if (target[0] != 0 || target[1] != 0) {
		if (target_corrupt_flag) {
			return -1;
		}
		target_corrupt_flag = 1;
		printf("Received corrupt target from Sia\n");
		printf("Usually this resolves itself within a minute or so\n");
		printf("If it happens frequently trying increasing seconds per iteration\n");
		printf("e.g. \"./gpu-miner -s 3 -c 200\"\n");
		printf("Waiting for problem to be resolved...");
		fflush(stdout);
	}
	target_corrupt_flag = 0;

	// By doing a bunch of low intensity calls, we prevent freezing
	// By splitting them up inside this function, we also avoid calling
	// get_block_for_work too often
	for (i = 0; i < cycles_per_iter; i++) {
		// Kernel sets nonce most significant bits, so we'll change least significant here
		blockHeader[38] = i / 256;
		blockHeader[39] = i % 256;

		// Copy input data to the memory buffer
		ret = clEnqueueWriteBuffer(command_queue, blockHeadermobj, CL_TRUE, 0, 80 * sizeof(uint8_t), blockHeader, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to blockHeadermobj buffer: %d\n", ret); exit(1); }
		ret = clEnqueueWriteBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to headerHashmobj buffer: %d\n", ret); exit(1); }
		ret = clEnqueueWriteBuffer(command_queue, targmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), target, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); exit(1); }

		// Execute OpenCL kernel as data parallel
		size_t local_item_size = 256;
		items_per_iter -= items_per_iter % 256;
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &items_per_iter, &local_item_size, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to start kernel: %d\n", ret); exit(1); }

		// Copy result to host
		ret = clEnqueueReadBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to read header hash from buffer: %d\n", ret); exit(1); }
		ret = clEnqueueReadBuffer(command_queue, nonceOutmobj, CL_TRUE, 0, 8 * sizeof(uint8_t), nonceOut, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to read nonce from buffer: %d\n", ret); exit(1); }

		// Did we find one?
		if (memcmp(headerHash, target, 8) < 0) {
			// Copy nonce to header.
			memcpy(blockHeader+32, nonceOut, 8);
			submit_header(curl, blockHeader);
			blocks_mined++;
			return -1;
		}
	}

	// Hashrate is inaccurate if a block was found
	#ifdef __linux__
	clock_gettime(CLOCK_REALTIME, &end);
	double nanosecondsElapsed = 1e9 * (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec);
	double run_time_seconds = nanosecondsElapsed * 1e-9;
	#else
	double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
	#endif
	double hash_rate = cycles_per_iter * items_per_iter / (run_time_seconds*1000000);

	return hash_rate;
}

int main(int argc, char *argv[]) {
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;

	int i, c, cycles_per_iter;
	char *port_number;
	double hash_rate, seconds_per_iter;
	size_t items_per_iter = 256*256*16;

	// parse args
	cycles_per_iter = 15;
	seconds_per_iter = 1.0;
	port_number = "9980";
	while ( (c = getopt(argc, argv, "hc:s:p:")) != -1) {
		switch (c) {
		case 'h':
			printf("\nUsage:\n\n");
			printf("\t c - cycles per iter: Number of workloads hashing gets split into each iteration\n");
			printf("\t\tIncrease this if your computer is freezing or locking up\n");
			printf("\n");
			printf("\t s - seconds per iter: Time between Sia API calls and hash rate updates\n");
			printf("\t\tIncrease this if your miner is receiving invalid targets\n");
			printf("\n");
			exit(0);
			break;
		case 'c':
			sscanf(optarg, "%d", &cycles_per_iter);
			if (cycles_per_iter < 1 || cycles_per_iter > 1000) {
				printf("Cycles per iter must be at least 1 and no more than 1000\n");
				exit(1);
			}
			break;
		case 's':
			sscanf(optarg, "%lf", &seconds_per_iter);
			break;
		case 'p':
			port_number = strdup(optarg);
			break;
		}
	}

	// Set siad URL
	set_port(port_number);

	// Use curl to communicate with siad
	curl = curl_easy_init();

	// Load kernel source file
	printf("Initializing...");
	fflush(stdout);
	FILE *fp;
	const char fileName[] = "./gpu-miner.cl";
	size_t source_size;
	char *source_str;
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get Platform/Device Information
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) { printf("failed to get platform IDs: %d\n", ret); exit(1); }
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (ret != CL_SUCCESS) { printf("failed to get Device IDs: %d\n", ret); exit(1); }

	// Create OpenCL Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create Buffer Objects
	blockHeadermobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 80 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create blockHeadermobj buffer: %d\n", ret); exit(1); }
	headerHashmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create targmobj buffer: %d\n", ret); exit(1); }
	targmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create targmobj buffer: %d\n", ret); exit(1); }
	nonceOutmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create nonceOutmobj buffer: %d\n", ret); exit(1); }

	// Create kernel program from source file
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if (ret != CL_SUCCESS) { printf("failed to crate program with source: %d\n", ret); exit(1); }
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		// Print information about why the build failed
		// This code is from StackOverflow
		size_t len;
		char buffer[204800];
		cl_build_status bldstatus;
		printf("\nError %d: Failed to build program executable [ ]\n", ret);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void *)&bldstatus, &len);
		if (ret != CL_SUCCESS) {
			printf("Build Status error %d\n", ret);
			exit(1);
		}
		if (bldstatus == CL_BUILD_SUCCESS) printf("Build Status: CL_BUILD_SUCCESS\n");
		if (bldstatus == CL_BUILD_NONE) printf("Build Status: CL_BUILD_NONE\n");
		if (bldstatus == CL_BUILD_ERROR) printf("Build Status: CL_BUILD_ERROR\n");
		if (bldstatus == CL_BUILD_IN_PROGRESS) printf("Build Status: CL_BUILD_IN_PROGRESS\n");
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
		if (ret != CL_SUCCESS) {
			printf("Build Options error %d\n", ret);
			exit(1);
		}
		printf("Build Options: %s\n", buffer);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		if (ret != CL_SUCCESS) {
			printf("Build Log error %d\n", ret);
			exit(1);
		}
		printf("Build Log:\n%s\n", buffer);
		exit(1);
	}

	// Create data parallel OpenCL kernel
	kernel = clCreateKernel(program, "nonceGrind", &ret);

	// Set OpenCL kernel arguments
	void *args[] = { &blockHeadermobj, &headerHashmobj, &targmobj, &nonceOutmobj };
	for (i = 0; i < 4; i++) {
		ret = clSetKernelArg(kernel, i, sizeof(cl_mem), args[i]);
		if (ret != CL_SUCCESS) {
			printf("failed to set kernel arg %d (error code %d)\n", i, ret);
			exit(1);
		}
	}
	printf("\n");

	// Make each iteration take about 1 second
	#ifdef __linux__
	struct timespec begin, end;
	clock_gettime(CLOCK_REALTIME, &begin);
	#else
	clock_t startTime = clock();
	#endif
	grindNonces(items_per_iter, 1);
	#ifdef __linux__
	clock_gettime(CLOCK_REALTIME, &end);

	double nanosecondsElapsed = 1e9 * (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec);
	double run_time_seconds = nanosecondsElapsed * 1e-9;
	#else
	double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
	#endif
	items_per_iter *= (seconds_per_iter / run_time_seconds) / cycles_per_iter;

	// Grind nonces until SIGINT
	signal(SIGINT, quitSignal);
	while (!quit) {
		// Repeat until no block is found
		do {
			hash_rate = grindNonces(items_per_iter, cycles_per_iter);
		} while (hash_rate == -1);

		if (!quit) {
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
	}

	// Finalization
	ret = clFlush(command_queue);   
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(blockHeadermobj);
	ret = clReleaseMemObject(headerHashmobj);
	ret = clReleaseMemObject(targmobj);
	ret = clReleaseMemObject(nonceOutmobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);	

	curl_easy_cleanup(curl);

	free(source_str);

	return 0;
}
