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

cl_command_queue command_queue = NULL;
cl_mem blockHeadermobj = NULL;
cl_mem headerHashmobj = NULL;
cl_mem targmobj = NULL;
cl_mem nonceOutmobj = NULL;
cl_mem nonceOutLockmobj = NULL;
cl_mem iter_per_threadmobj = NULL;
cl_kernel kernel = NULL;
cl_int ret;

CURL *curl;

unsigned int blocks_mined = 0;


// Perform global_item_size * iter_per_thread hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
double grindNonces(size_t global_item_size, size_t iter_per_thread) {
	uint8_t blockHeader[80];
	uint8_t headerHash[32];
	uint8_t target[32];
	uint8_t nonceOut[8]; // This is where the nonce that gets a low enough hash will be stored
	uint8_t nonceOutLock = 0;

	int i;
	for (i = 0; i < 8; i++) {
		nonceOut[i] = 0;
	}

	// Max out hash
	for (i = 0; i < 32; i++) {
		headerHash[i] = 255;
	}

	// Store block from siad
	uint8_t *block;
	size_t blocklen = 0;

	// Get new block header and target
	get_block_for_work(curl, target, blockHeader, &block, &blocklen);

	// Start timing this iteration
	clock_t startTime = clock();

	// Copy input data to the memory buffer
	ret = clEnqueueWriteBuffer(command_queue, blockHeadermobj, CL_TRUE, 0, 80 * sizeof(uint8_t), blockHeader, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to write to blockHeadermobj buffer: %d\n", ret); exit(1); }
	ret = clEnqueueWriteBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); exit(1); }
	ret = clEnqueueWriteBuffer(command_queue, targmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), target, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); exit(1); }
	ret = clEnqueueWriteBuffer(command_queue, nonceOutLockmobj, CL_TRUE, 0, sizeof(uint8_t), &nonceOutLock, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to write to nonceOutLockmobj buffer: %d\n", ret); exit(1); }
	ret = clEnqueueWriteBuffer(command_queue, iter_per_threadmobj, CL_TRUE, 0, sizeof(uint32_t), &iter_per_thread, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to write to targmobj buffer: %d\n", ret); exit(1); }

	// Execute OpenCL kernel as data parallel
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to start kernel: %d\n", ret); exit(1); }

	// Copy result to host
	ret = clEnqueueReadBuffer(command_queue, headerHashmobj, CL_TRUE, 0, 32 * sizeof(uint8_t), headerHash, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to read header hash from buffer: %d\n", ret); exit(1); }
	ret = clEnqueueReadBuffer(command_queue, nonceOutmobj, CL_TRUE, 0, 8 * sizeof(uint8_t), nonceOut, 0, NULL, NULL);
	if (ret != CL_SUCCESS) { printf("failed to read nonce from buffer: %d\n", ret); exit(1); }

	// Did we find one?
	i = 0;
	while (target[i] == headerHash[i]) {
		i++;
	}
	if (headerHash[i] < target[i]) {
		// Copy nonce to block
		for (i = 0; i < 8; i++) {
			block[i + 32] = nonceOut[i];
		}

		submit_block(curl, block, blocklen);
		blocks_mined++;
	} else {
		// Hashrate is inaccurate if a block was found
		double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
		double hash_rate = (iter_per_thread*global_item_size) / (run_time_seconds*1000000);
		// TODO: Print est time until next block (target difficulty / hashrate
		return hash_rate;
	}
	return -1;
}

int main() {   
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
 
	int i;
	int max_compute_units;
	size_t global_item_size = 1;
	size_t iter_per_thread = 256 * 16; // This must be a multiple of 256 and no more than 256 * 256

	// Use curl to communicate with siad
	curl = curl_easy_init();

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
	if (ret != CL_SUCCESS) { printf("failed to get platform IDs: %d\n", ret); exit(1); }
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (ret != CL_SUCCESS) { printf("failed to get Device IDs: %d\n", ret); exit(1); }
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &max_compute_units, NULL);
	if (ret != CL_SUCCESS) { printf("failed to get device max compute units: %d\n", ret); exit(1); }
	printf("Device max compute units:\t%d\n", max_compute_units);

	// Create OpenCL Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create Buffer Objects
	blockHeadermobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 80 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create blockHeadermobj buffer: %d\n", ret); exit(1); }
	headerHashmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create targmobj buffer: %d\n", ret); exit(1); }
	targmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 32 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create targmobj buffer: %d\n", ret); exit(1); }
	nonceOutmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create nonceOutmobj buffer: %d\n", ret); exit(1); }
	nonceOutLockmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create nonceOutmobj buffer: %d\n", ret); exit(1); }
	iter_per_threadmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create iter_per_threadmobj buffer: %d\n", ret); exit(1); }

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
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&blockHeadermobj);
	if (ret != CL_SUCCESS) { printf("failed to set first kernel arg: \n"); exit(1); }
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&headerHashmobj);
	if (ret != CL_SUCCESS) { printf("failed to set fifth kernel arg: \n"); exit(1); }
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&targmobj);
	if (ret != CL_SUCCESS) { printf("failed to set third kernel arg: \n"); exit(1); }
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&nonceOutmobj);
	if (ret != CL_SUCCESS) { printf("failed to set second kernel arg: \n"); exit(1); }
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&nonceOutLockmobj);
	if (ret != CL_SUCCESS) { printf("failed to set fourth kernel arg: \n"); exit(1); }
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&iter_per_threadmobj);
	if (ret != CL_SUCCESS) { printf("failed to set fourth kernel arg: \n"); exit(1); }

	// Rough scan for 'optimal' thread count
	double hash_rate, prev_hash_rate = 0;
	global_item_size = 192;
	iter_per_thread = 16 * 256;
	while(global_item_size < (256*256)/2) {
		global_item_size *= 2;

		// Make each iteration take about 3 seconds
		clock_t startTime = clock();
		double temp = grindNonces(global_item_size, iter_per_thread);
		double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
		iter_per_thread *= 3 / run_time_seconds;

		while (temp == -1) {
			// Repeat until no block is found
			temp = grindNonces(global_item_size, iter_per_thread);
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
		hash_rate = temp;
		printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
		fflush(stdout);
		prev_hash_rate = hash_rate;
	}
	printf("\rRough search found %zd threads to be the best at %.3f MH/s\n", global_item_size/2, prev_hash_rate);
	fflush(stdout);

	// Now we know the optimal is betweem global_item_size and global_item_size / 2
	// Scan intermediate 16 values and pick the highest
	int step_size = (global_item_size - global_item_size / 2) / 16;
	double best_hash_rate = prev_hash_rate;
	size_t best_item_size = global_item_size / 2;
	for (i = 0; i <= 16; i++) {
		global_item_size -= step_size;

		// Make each iteration take about 3 seconds
		clock_t startTime = clock();
		double temp = grindNonces(global_item_size, iter_per_thread);
		double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
		iter_per_thread *= 3 / run_time_seconds;

		while (temp == -1) {
			// Repeat until no block is found
			temp = grindNonces(global_item_size, iter_per_thread);
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
		hash_rate = temp;
		printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
		fflush(stdout);
		if (hash_rate > best_hash_rate) {
			best_hash_rate = hash_rate;
			best_item_size = global_item_size;
		}
	}
	global_item_size = best_item_size;
	printf("\rFine search found %zd threads to be the best at %.3f MH/s\n", global_item_size, best_hash_rate);
	fflush(stdout);

	// Make each iteration take about 3 seconds
	clock_t startTime = clock();
	grindNonces(global_item_size, iter_per_thread);
	double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
	iter_per_thread *= 3 / run_time_seconds;

	// Grind nonces endlessly using
	while (1) {
		double temp = grindNonces(global_item_size, iter_per_thread);
		while (temp == -1) {
			// Repeat until no block is found
			temp = grindNonces(global_item_size, iter_per_thread);
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
		hash_rate = temp;
		printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
		fflush(stdout);
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
	ret = clReleaseMemObject(iter_per_threadmobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);	
 
 	curl_easy_cleanup(curl);
 	
	free(source_str);
 
	return 0;
}
