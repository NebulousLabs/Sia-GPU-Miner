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

// Minimum intensity being less than 8 may break things
#define MIN_INTENSITY		8
#define MAX_INTENSITY		32
#define DEFAULT_INTENSITY	16

// TODO: Might wanna establish a min/max for this, too...
#define DEFAULT_CPI			3

#define MAX_SOURCE_SIZE 	(0x200000)

cl_command_queue command_queue = NULL;
cl_mem blockHeadermobj = NULL;
cl_mem headerHashmobj = NULL;
cl_mem targmobj = NULL;
cl_mem nonceOutmobj = NULL;
cl_kernel kernel = NULL;
cl_int ret;

CURL *curl;

unsigned int blocks_mined = 0, Intensity = DEFAULT_INTENSITY;
static volatile int quit = 0;
int target_corrupt_flag = 0;

void quitSignal(int __unused) {
	quit = 1;
	printf("\nCaught deadly signal, quitting...\n");
}

// Perform (2^Intensity) * cycles_per_iter hashes
// Return -1 if a block is found
// Else return the hashrate in MH/s
double grindNonces(int cycles_per_iter) {
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
	size_t GlobalSize = 1 << Intensity;
	
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
		// Note that minimum intensity is 8, making the local
		// worksize always divisible by the global worksize
		size_t local_item_size = 256;
		
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &GlobalSize, &local_item_size, 0, NULL, NULL);
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
	double hash_rate = cycles_per_iter * GlobalSize / (run_time_seconds*1000000);

	return hash_rate;
}

void SelectOCLDevice(cl_platform_id *OCLPlatform, cl_device_id *OCLDevice, cl_uint PlatformIdx, cl_uint DeviceIdx)
{
	cl_uint PlatformCount, DeviceCount;
	cl_platform_id *AllPlatforms;
	cl_device_id *AllDevices;
	cl_int ret;
	
	ret = clGetPlatformIDs(0, NULL, &PlatformCount);
	if(ret != CL_SUCCESS)
	{
		printf("Failed to get number of OpenCL platforms with error code %d (clGetPlatformIDs).\n", ret);
		exit(1);
	}
	
	// If we don't exit here, the default platform ID chosen MUST be valid; it's zero.
	// I return 0, because this isn't an error - there is simply nothing to do.
	if(!PlatformCount)
	{
		printf("OpenCL is reporting no platforms available on the system. Nothing to do.\n");
		exit(0);
	}
	
	// Since the number of platforms returned is the number of indexes plus one,
	// the default platform ID (zero), must exist. User may still specify something
	// invalid, however, so check it.
	if(PlatformCount <= PlatformIdx)
	{
		printf("Platform selected (%u) is the same as, or higher than, the number ", PlatformIdx);
		printf("of platforms reported to exist by OpenCL on this system (%u). ", PlatformCount);
		printf("Remember that the first platform has index 0!\n");
		exit(1);
	}
	
	AllPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * PlatformCount);
	
	ret = clGetPlatformIDs(PlatformCount, AllPlatforms, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("Failed to retrieve OpenCL platform IDs with error code %d (clGetPlatformIDs).\n", ret);
		exit(1);
	}
	
	// Now fetch device ID list for this platform similarly to the fetch for the platform IDs.
	// PlatformIdx has been verified to be within bounds.
	ret = clGetDeviceIDs(AllPlatforms[PlatformIdx], CL_DEVICE_TYPE_GPU, 0, NULL, &DeviceCount);
	if(ret != CL_SUCCESS)
	{
		printf("Failed to get number of OpenCL devices with error code %d (clGetDeviceIDs).\n", ret);
		free(AllPlatforms);
		exit(1);
	}
	
	// If we have no devices, indicate this to the user
	if(!DeviceCount)
	{
		printf("OpenCL is reporting no GPU devices available for chosen platform. Nothing to do.\n");
		free(AllPlatforms);
		exit(0);
	}
	
	// Check that the device we've been asked to get does, in fact, exist...
	if(DeviceCount <= DeviceIdx)
	{
		printf("Device selected (%u) is the same as, or higher than, the number ", DeviceIdx);
		printf("of GPU devices reported to exist by OpenCL on the current platform (%u). ", DeviceCount); 
		printf("Remember that the first device has index 0!\n");
		free(AllPlatforms);
		exit(1);
	}
	
	AllDevices = (cl_device_id *)malloc(sizeof(cl_device_id) * DeviceCount);
	
	ret = clGetDeviceIDs(AllPlatforms[PlatformIdx], CL_DEVICE_TYPE_GPU, DeviceCount, AllDevices, NULL);
	if(ret != CL_SUCCESS)
	{
		printf("Failed to retrieve OpenCL device IDs for selected platform with error code %d (clGetDeviceIDs).\n", ret);
		free(AllPlatforms);
		free(AllDevices);
		exit(1);
	}
	
	// Done. Return the platform ID and device ID object desired, free lists, and return.
	*OCLPlatform = AllPlatforms[PlatformIdx];
	*OCLDevice = AllDevices[DeviceIdx];
}
	
int main(int argc, char *argv[]) {
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_uint PlatformIdx = 0, DeviceIdx = 0;
	int i, c;
	unsigned cycles_per_iter;
	char *port_number;
	double hash_rate;

	// parse args
	cycles_per_iter = DEFAULT_CPI;
	port_number = "9980";
	while ( (c = getopt(argc, argv, "hI:p:d:C:P:")) != -1) {
		switch (c) {
		case 'h':
			printf("\nUsage:\n\n");
			printf("\t I - intensity: This is the amount of work sent to the GPU in one batch.\n");
			printf("\t\tInterpretation is 2^intensity; the default is %d. Lower if GPU crashes or\n");
			printf("\t\tif more desktop interactivity is desired. Raising it may improve performance.\n", DEFAULT_INTENSITY);
			printf("\n");
			printf("\t p - OpenCL platform ID: Just what it says on the tin. If you're finding no GPUs,\n");
			printf("\t\tyet you're sure they exist, try a value other than 0, like 1, or 2. Default is 0.\n");
			printf("\n");
			printf("\t d - OpenCL device ID: Self-explanatory; it's the GPU index. Note that different\n");
			printf("\t\tOpenCL platforms will likely have different devices available. Default is 0.\n");
			printf("\n");
			printf("\t C - cycles per iter: Number of kernel executions between Sia API calls and hash rate updates\n");
			printf("\t\tIncrease this if your miner is receiving invalid targets. Default is %ud.\n", DEFAULT_CPI);
			printf("\n");
			exit(0);
			break;
		case 'I':
			Intensity = strtoul(optarg, NULL, 10);		// Returns zero on error
			
			if(!Intensity || (Intensity < MIN_INTENSITY) || (Intensity > MAX_INTENSITY))
			{
				printf("Intensity either set to zero, or invalid. Default will be used.\n");
				printf("Note that the minimum intensity is %d, and the maximum is %d.\n", MIN_INTENSITY, MAX_INTENSITY);
				Intensity = DEFAULT_INTENSITY;
			}
			break;
		case 'p':
			// Again, zero return on error. Default is zero.
			// I don't see  a problem here.
			PlatformIdx = strtoul(optarg, NULL, 10);
			break;
		case 'd':
			// See comment for previous option.
			DeviceIdx = strtoul(optarg, NULL, 10);
			break;
		case 'C':
			sscanf(optarg, "%ud", &cycles_per_iter);
			if(!cycles_per_iter) cycles_per_iter = DEFAULT_CPI;
			break;
		case 'P':
			port_number = strdup(optarg);
			break;
		}
	}

	// Set siad URL
	set_port(port_number);

	// Use curl to communicate with siad
	curl = curl_easy_init();

	// Load kernel source file
	printf("Initializing...\n");
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
	/*ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) { printf("failed to get platform IDs: %d\n", ret); exit(1); }
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (ret != CL_SUCCESS) { printf("failed to get Device IDs: %d\n", ret); exit(1); }
	*/
	
	SelectOCLDevice(&platform_id, &device_id, PlatformIdx, DeviceIdx);
	
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

	// Grind nonces until SIGINT
	signal(SIGINT, quitSignal);
	while (!quit) {
		// Repeat until no block is found
		do {
			hash_rate = grindNonces(cycles_per_iter);
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
