#include <time.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#include "network.h"
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>  
#endif
 
#define MAX_SOURCE_SIZE (0x200000)
#define THREADS_PER_COMPUTE_UNIT 192
#define THREAD_MULT 16
/* GPU does MAX_COMPUTE_UNITS * THREADS_PERCOMPUTE_UNIT * THREAD_MULT threads.
 * Maxes out at 256 * 256 threads to make nonce grinding simpler for now.
 * Using such a large number of threads helps ensure that every core stays busy.
 * It can be faster to have less threads with each doing more work, but only if you know the right number
 * of threads for your GPU. If you get this number wrong, it can severely impact performance.
 * Using this method usually gets 85-95% hashing power out of the GPU.
 * Using a lower, non-optimized number of threads can result in as low as 60% hashing power.
 * TODO: Write code that finds the 'optimal' thread count on host GPU to get 95-100% hashing power
 */
 
int main() {   
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem blockHeadermobj = NULL;
	cl_mem headerHashmobj = NULL;
	cl_mem targmobj = NULL;
	cl_mem nonceOutmobj = NULL;	
	cl_mem nonceOutLockmobj = NULL;	
	cl_mem numItersPerThreadmobj = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;	
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	int max_compute_units;

	// Use curl to communicate with siad
	CURL *curl = curl_easy_init();
 
 	// Initialize the kernel's input data.
	int i;
	uint8_t blockHeader[80];
	uint8_t headerHash[32];
	uint8_t target[32];
	uint8_t nonceOut[8]; // This is where the nonce that gets a low enough hash will be stored
	uint8_t nonceOutLock = 0;
	uint32_t numItersPerThread = 256 * 16; // This must be a multiple of 256 and no more than 256 * 256

	// Store block from siad
	uint8_t *block;
	size_t blocklen = 0;

	for (i = 0; i < 8; i++)
		nonceOut[i] = 0;

	// Load kernel source file
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
	if (ret != CL_SUCCESS) { printf("failed to get platform IDs: %d\n", ret); return -1; }
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (ret != CL_SUCCESS) { printf("failed to get Device IDs: %d\n", ret); return -1; }
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &max_compute_units, NULL);
	if (ret != CL_SUCCESS) { printf("failed to get device max compute units: %d\n", ret); return -1; }
	printf("Device max compute units:\t%d\n", max_compute_units);

	// Set number of threads to run
	size_t global_item_size = max_compute_units * THREADS_PER_COMPUTE_UNIT * THREAD_MULT;
	if (global_item_size > 65536)
		global_item_size = 65536;

	// Create OpenCL Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create Buffer Objects
	blockHeadermobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 80 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create blockHeadermobj buffer: %d\n", ret); return -1; }
	headerHashmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create targmobj buffer: %d\n", ret); return -1; }
	targmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create targmobj buffer: %d\n", ret); return -1; }
	nonceOutmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create nonceOutmobj buffer: %d\n", ret); return -1; }
	nonceOutLockmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create nonceOutmobj buffer: %d\n", ret); return -1; }
	numItersPerThreadmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create numItersPerThreadmobj buffer: %d\n", ret); return -1; }

	// Create kernel program from source file
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if (ret != CL_SUCCESS) printf("failed to build with source: %d\n", ret);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		// Print information about why the build failed
		// This code is from StackOverflow
		size_t len;
		char buffer[204800];
		cl_build_status bldstatus;
		printf("\nError %d: Failed to build program executable [ ]\n", ret);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void *)&bldstatus, &len);
		if (ret != CL_SUCCESS)
		{
			printf("Build Status error %d\n", ret);
			exit(1);
		}
		if (bldstatus == CL_BUILD_SUCCESS) printf("Build Status: CL_BUILD_SUCCESS\n");
		if (bldstatus == CL_BUILD_NONE) printf("Build Status: CL_BUILD_NONE\n");
		if (bldstatus == CL_BUILD_ERROR) printf("Build Status: CL_BUILD_ERROR\n");
		if (bldstatus == CL_BUILD_IN_PROGRESS) printf("Build Status: CL_BUILD_IN_PROGRESS\n");
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
		if (ret != CL_SUCCESS)
		{
			printf("Build Options error %d\n", ret);
			exit(1);
		}
		printf("Build Options: %s\n", buffer);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		if (ret != CL_SUCCESS)
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
	if (ret != CL_SUCCESS) { printf("failed to set first kernel arg: \n"); return -1; }
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&headerHashmobj);
	if (ret != CL_SUCCESS) { printf("failed to set fifth kernel arg: \n"); return -1; }
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&targmobj);
	if (ret != CL_SUCCESS) { printf("failed to set third kernel arg: \n"); return -1; }
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&nonceOutmobj);
	if (ret != CL_SUCCESS) { printf("failed to set second kernel arg: \n"); return -1; }
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&nonceOutLockmobj);
	if (ret != CL_SUCCESS) { printf("failed to set fourth kernel arg: \n"); return -1; }
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&numItersPerThreadmobj);
	if (ret != CL_SUCCESS) { printf("failed to set fourth kernel arg: \n"); return -1; }

	// Mine blocks until program is interrupted
	// Each iteration of the loop should take 1-3 seconds
	while (1) {
		// Start timing this iteration
		clock_t startTime = clock();

		// Get new block header and target
		get_block_for_work(curl, target, blockHeader, &block, &blocklen);

		// Reset hash
		for (i = 0; i < 32; i++)
			headerHash[i] = 255;

		nonceOutLock = 0;

		// Copy input data to the memory buffer
		ret = clEnqueueWriteBuffer(command_queue, blockHeadermobj, CL_TRUE, 0, 80 * sizeof(uint8_t), blockHeader, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to blockHeadermobj buffer: %d\n", ret); return -1; }
		ret = clEnqueueWriteBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); return -1; }
		ret = clEnqueueWriteBuffer(command_queue, targmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), target, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); return -1; }
		ret = clEnqueueWriteBuffer(command_queue, nonceOutLockmobj, CL_TRUE, 0, sizeof(uint8_t), &nonceOutLock, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to nonceOutLockmobj buffer: %d\n", ret); return -1; }
		ret = clEnqueueWriteBuffer(command_queue, numItersPerThreadmobj, CL_TRUE, 0, sizeof(uint32_t), &numItersPerThread, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); return -1; }

		// Execute OpenCL kernel as data parallel
		printf("Starting %zd threads.\n", global_item_size);
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to start kernel: %d\n", ret); return -1; }

		// Copy result to host
		ret = clEnqueueReadBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to read header hash from buffer: %d\n", ret); return -1; }
		ret = clEnqueueReadBuffer(command_queue, nonceOutmobj, CL_TRUE, 0, 8 * sizeof(uint8_t), nonceOut, 0, NULL, NULL);
		if (ret != CL_SUCCESS) { printf("failed to read nonce from buffer: %d\n", ret); return -1; }

		// Did we find one?
		i = 0;
		while (target[i] == headerHash[i])
			i++;
		if (headerHash[i] < target[i]) {
			// Display some info about the hash that was found
			printf("Thread %u found a good hash!\n", nonceOut[0] * 256 + nonceOut[1]);

			printf("Header: [");
			for (i = 0; i < 10; i++) {
				printf("%u ", blockHeader[i]);
			}
			printf("... %u]\n", blockHeader[79]);

			printf("Hash: [");
			for (i = 0; i < 10; i++) {
				printf("%u ", headerHash[i]);
			}
			printf("... %u]\n", headerHash[31]);

			printf("Nonce: [");
			for (i = 0; i < 7; i++) {
				printf("%u ", nonceOut[i]);
			}
			printf("%u]\n", nonceOut[7]);
			
			// Copy nonce to block
			for (i = 0; i < 8; i++)
				block[i + 32] = nonceOut[i];

			submit_block(curl, block, blocklen);
			printf("\n");
		} else {
			printf("No hash was found. Fetching new block.\n");
			// Hashrate is inaccurate if a block was found
			double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
			printf("Mined for %.2f seconds at %.3f MH/s\n\n", run_time_seconds, (numItersPerThread*global_item_size) / (run_time_seconds*1000000));
			// TODO: Print est time until next block (target difficulty / hashrate
		}
	}
	
	// Finalization
	ret = clFlush(command_queue);   
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(blockHeadermobj);
	ret = clReleaseMemObject(targmobj);
	ret = clReleaseMemObject(nonceOutmobj);
	ret = clReleaseMemObject(nonceOutLockmobj);
	ret = clReleaseMemObject(numItersPerThreadmobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);	
 
 	curl_easy_cleanup(curl);
 	
	free(source_str);
 
	return 0;
}
