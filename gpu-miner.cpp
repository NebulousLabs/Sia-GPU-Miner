#ifdef __linux__
#define _GNU_SOURCE
#define _POSIX_SOURCE
#include <sys/time.h>
#endif

#include <ctime>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
using namespace std;

#include "network.h"

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
	ret = clEnqueueWriteBuffer(command_queue, blockHeadermobj, CL_TRUE, 0, 80 * sizeof(uint8_t), blockHeader, 0, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("failed to write to blockHeadermobj buffer: %d\n", ret); exit(1);
	}
	ret = clEnqueueWriteBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("failed to write to headerHashmobj buffer: %d\n", ret); exit(1);
	}
	ret = clEnqueueWriteBuffer(command_queue, targmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), target, 0, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("failed to write to targmobj buffer: %d\n", ret); exit(1);
	}

	// Execute OpenCL kernel as data parallel
	size_t local_item_size = 256;
	global_item_size -= global_item_size % 256;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("failed to start kernel: %d\n", ret); exit(1);
	}

	// Copy result to host
	ret = clEnqueueReadBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("failed to read header hash from buffer: %d\n", ret); exit(1);
	}
	ret = clEnqueueReadBuffer(command_queue, nonceOutmobj, CL_TRUE, 0, 8 * sizeof(uint8_t), nonceOut, 0, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("failed to read nonce from buffer: %d\n", ret); exit(1);
	}

	// Did we find one?
	i = 0;
	while(target[i] == headerHash[i])
	{
		i++;
	}
	if(headerHash[i] < target[i])
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
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;

	int i;
	size_t global_item_size = 1;

	// Use curl to communicate with siad
	curl = curl_easy_init();

	// Load kernel source file
	FILE *fp;
	const char fileName[] = "./gpu-miner.cl";
	size_t source_size;
	char *source_str;
	fp = fopen(fileName, "r");
	if(!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get Platform/Device Information
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if(ret != CL_SUCCESS)
	{
		printf("failed to get platform IDs: %d\n", ret); exit(1);
	}
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if(ret != CL_SUCCESS)
	{
		printf("failed to get Device IDs: %d\n", ret); exit(1);
	}

	// Create OpenCL Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create Buffer Objects
	blockHeadermobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 80 * sizeof(uint8_t), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printf("failed to create blockHeadermobj buffer: %d\n", ret); exit(1);
	}
	headerHashmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(uint8_t), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printf("failed to create targmobj buffer: %d\n", ret); exit(1);
	}
	targmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 32 * sizeof(uint8_t), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printf("failed to create targmobj buffer: %d\n", ret); exit(1);
	}
	nonceOutmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(uint8_t), NULL, &ret);
	if(ret != CL_SUCCESS)
	{
		printf("failed to create nonceOutmobj buffer: %d\n", ret); exit(1);
	}

	// Create kernel program from source file
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if(ret != CL_SUCCESS)
	{
		printf("failed to crate program with source: %d\n", ret); exit(1);
	}
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(ret != CL_SUCCESS)
	{
		// Print information about why the build failed
		// This code is from StackOverflow
		size_t len;
		char buffer[204800];
		cl_build_status bldstatus;
		printf("\nError %d: Failed to build program executable [ ]\n", ret);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void *)&bldstatus, &len);
		if(ret != CL_SUCCESS)
		{
			printf("Build Status error %d\n", ret);
			exit(1);
		}
		if(bldstatus == CL_BUILD_SUCCESS) printf("Build Status: CL_BUILD_SUCCESS\n");
		if(bldstatus == CL_BUILD_NONE) printf("Build Status: CL_BUILD_NONE\n");
		if(bldstatus == CL_BUILD_ERROR) printf("Build Status: CL_BUILD_ERROR\n");
		if(bldstatus == CL_BUILD_IN_PROGRESS) printf("Build Status: CL_BUILD_IN_PROGRESS\n");
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
		if(ret != CL_SUCCESS)
		{
			printf("Build Options error %d\n", ret);
			exit(1);
		}
		printf("Build Options: %s\n", buffer);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		if(ret != CL_SUCCESS)
		{
			printf("Build Log error %d\n", ret);
			exit(1);
		}
		printf("Build Log:\n%s\n", buffer);
		exit(1);
	}

	// Create data parallel OpenCL kernel
	kernel = clCreateKernel(program, "nonceGrind", &ret);

	// Set OpenCL kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&blockHeadermobj);
	if(ret != CL_SUCCESS)
	{
		printf("failed to set first kernel arg: \n"); exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&headerHashmobj);
	if(ret != CL_SUCCESS)
	{
		printf("failed to set fifth kernel arg: \n"); exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&targmobj);
	if(ret != CL_SUCCESS)
	{
		printf("failed to set third kernel arg: \n"); exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&nonceOutmobj);
	if(ret != CL_SUCCESS)
	{
		printf("failed to set second kernel arg: \n"); exit(1);
	}

	double hash_rate;
	global_item_size = 256 * 256 * 16;

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
