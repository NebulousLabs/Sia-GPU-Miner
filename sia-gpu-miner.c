/*
 * A cross-platform GPU miner built for Sia
 * using OpenCL for interacting with the graphics card
 * and libcurl for interacting with the Sia daemon.
 */

// Some linux distros need a different timer.
#ifdef __linux__
#define _GNU_SOURCE
#define _POSIX_SOURCE
#include <sys/time.h>
#endif

// OpenCL headers are different for Apple.
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <time.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include "network.h"

// 2^intensity hashes are calculated each time the kernel is called
// Minimum of 2^8 (256) because our default local_item_size is 256
// global_item_size (2^intensity) must be a multiple of local_item_size
// Max of 2^32 so that people can't send an hour of work to the GPU at one time
#define MIN_INTENSITY		8
#define MAX_INTENSITY		32
#define DEFAULT_INTENSITY	16

// Number of times the GPU kernel is called between updating the command line text
#define MIN_CPI 		1     // Must do one call per update
#define MAX_CPI 		65536 // 2^16 is a slightly arbitrary max
#define DEFAULT_CPI		30

// The maximum size of the .cl file we read in and compile
#define MAX_SOURCE_SIZE 	(0x200000)

// Objects needed to call the kernel
// global namespace so our grindNonce function can access them
cl_command_queue command_queue = NULL;
cl_kernel kernel = NULL;
cl_int ret;

// mem objects for storing our kernel parameters
cl_mem blockHeadermobj = NULL;
cl_mem nonceOutmobj = NULL;

// More gobal variables the grindNonce needs to access
size_t local_item_size = 256; // Size of local work groups. 256 is usually optimal
unsigned int blocks_mined = 0;
unsigned int intensity = DEFAULT_INTENSITY;
static volatile int quit = 0;

// If we get a corrupt target, we want to remember so that if subsequent curl calls
// reutrn more corrupt targets, we don't spam the cmd line with errors
int target_corrupt_flag = 0;

// Set quit variable when SIGINT is received so we can do proper cleanup
void quitSignal(int unused) {
	(void)unused; // prevents clang from complaining about an unused variable.
	quit = 1;
	printf("\nCaught kill signal, quitting...\n");
}

// Given a number of cycles per iter, grind nonces will poll Sia for a block
// then do 2^intensity hashes cycles_per_iter times, checking for a successful
// hash each time
// Returns -1 if it finds a block, otherwise it returns the hash_rate of the GPU
double grindNonces(int cycles_per_iter) {
	// Start timing this iteration.
	#ifdef __linux__
	struct timespec begin, end;
	clock_gettime(CLOCK_REALTIME, &begin);
	#else
	clock_t startTime = clock();
	#endif

	uint8_t blockHeader[80];
	uint8_t target[32] = {255};
	uint8_t nonceOut[8] = {0};

	// Get new block header and target.
	if (get_header_for_work(target, blockHeader) != 0) {
		return -1;
	}

	// Check for target corruption.
	int i;
	if (target[0] != 0 || target[1] != 0) {
		if (target_corrupt_flag) {
			return -1;
		}
		target_corrupt_flag = 1;
		printf("Received corrupt target from Sia\n");
		printf("Waiting for problem to be resolved...");
		fflush(stdout);
		return -1;
	}
	target_corrupt_flag = 0;
	size_t global_item_size = 1 << intensity;

	// Copy target to header.
	for (i = 0; i < 8; i++) {
		blockHeader[i + 32] = target[7-i];
	}

	// By doing a bunch of low intensity calls, we prevent freezing
	// By splitting them up inside this function, we also avoid calling
	// get_block_for_work too often.
	for (i = 0; i < cycles_per_iter; i++) {
		// Offset global ids so that each loop call tries a different set of
		// hashes.
		size_t globalid_offset = i * global_item_size;

		// Copy input data to the memory buffer.
		ret = clEnqueueWriteBuffer(command_queue, blockHeadermobj, CL_TRUE, 0, 80 * sizeof(uint8_t), blockHeader, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("failed to write to blockHeadermobj buffer: %d\n", ret); exit(1);
		}
		ret = clEnqueueWriteBuffer(command_queue, nonceOutmobj, CL_TRUE, 0, 8 * sizeof(uint8_t), nonceOut, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("failed to write to targmobj buffer: %d\n", ret); exit(1);
		}

		// Run the kernel.
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, &globalid_offset, &global_item_size, &local_item_size, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("failed to start kernel: %d\n", ret); exit(1);
		}

		// Copy result to host and see if a block was found.
		ret = clEnqueueReadBuffer(command_queue, nonceOutmobj, CL_TRUE, 0, 8 * sizeof(uint8_t), nonceOut, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("failed to read nonce from buffer: %d\n", ret); exit(1);
		}
		if (nonceOut[0] != 0) {
			// Copy nonce to header.
			memcpy(blockHeader+32, nonceOut, 8);
			if (!submit_header(blockHeader)) {
				// Only count block if submit succeeded.
				blocks_mined++;
			}
			return -1;
		}
	}

	// Get the time elapsed this function.
	#ifdef __linux__
	clock_gettime(CLOCK_REALTIME, &end);
	double nsElapsed = 1e9 * (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_nsec - begin.tv_nsec);
	double run_time_seconds = nsElapsed * 1e-9;
	#else
	double run_time_seconds = (double)(clock() - startTime) / CLOCKS_PER_SEC;
	#endif

	// Calculate the hash rate of thie iteration.
	double hash_rate = cycles_per_iter * global_item_size / (run_time_seconds*1000000);
	return hash_rate;
}

// selectOCLDevice manages opencl device selection as requested by the command
// line arguments.
void selectOCLDevice(cl_platform_id *OCLPlatform, cl_device_id *OCLDevice, cl_uint platformid, cl_uint deviceidx) {
	cl_uint platformCount, deviceCount;
	cl_platform_id *platformids;
	cl_device_id *deviceids;
	cl_int ret;
	
	ret = clGetPlatformIDs(0, NULL, &platformCount);
	if(ret != CL_SUCCESS) {
		printf("Failed to get number of OpenCL platforms with error code %d (clGetPlatformIDs).\n", ret);
		exit(1);
	}
	
	// If we don't exit here, the default platform ID chosen MUST be valid; it's zero.
	// I return 0, because this isn't an error - there is simply nothing to do.
	if(!platformCount) {
		printf("OpenCL is reporting no platforms available on the system. Nothing to do.\n");
		exit(0);
	}
	
	// Since the number of platforms returned is the number of indexes plus one,
	// the default platform ID (zero), must exist. User may still specify something
	// invalid, however, so check it.
	if(platformCount <= platformid) {
		printf("Platform selected (%u) is the same as, or higher than, the number ", platformid);
		printf("of platforms reported to exist by OpenCL on this system (%u). ", platformCount);
		printf("Remember that the first platform has index 0!\n");
		exit(1);
	}
	
	platformids = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
	
	ret = clGetPlatformIDs(platformCount, platformids, NULL);
	if(ret != CL_SUCCESS) {
		printf("Failed to retrieve OpenCL platform IDs with error code %d (clGetPlatformIDs).\n", ret);
		exit(1);
	}
	
	// Now fetch device ID list for this platform similarly to the fetch for the platform IDs.
	// platformid has been verified to be within bounds.
	ret = clGetDeviceIDs(platformids[platformid], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
	if(ret != CL_SUCCESS) {
		printf("Failed to get number of OpenCL devices with error code %d (clGetDeviceIDs).\n", ret);
		free(platformids);
		exit(1);
	}
	
	// If we have no devices, indicate this to the user.
	if(!deviceCount) {
		printf("OpenCL is reporting no GPU devices available for chosen platform. Nothing to do.\n");
		free(platformids);
		exit(0);
	}
	
	// Check that the device we've been asked to get does, in fact, exist...
	if(deviceCount <= deviceidx) {
		printf("Device selected (%u) is the same as, or higher than, the number ", deviceidx);
		printf("of GPU devices reported to exist by OpenCL on the current platform (%u). ", deviceCount);
		printf("Remember that the first device has index 0!\n");
		free(platformids);
		exit(1);
	}
	
	deviceids = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
	
	ret = clGetDeviceIDs(platformids[platformid], CL_DEVICE_TYPE_GPU, deviceCount, deviceids, NULL);
	if(ret != CL_SUCCESS) {
		printf("Failed to retrieve OpenCL device IDs for selected platform with error code %d (clGetDeviceIDs).\n", ret);
		free(platformids);
		free(deviceids);
		exit(1);
	}
	
	// Done. Return the platform ID and device ID object desired, free lists, and return.
	*OCLPlatform = platformids[platformid];
	*OCLDevice = deviceids[deviceidx];
}

// printPlatformsAndDevices prints out a list of opencl platforms and devices
// that were found on the system.
void printPlatformsAndDevices() {
	cl_uint platformCount, deviceCount;
	cl_platform_id *platformids;
	cl_device_id *deviceids;
	cl_int ret;

	ret = clGetPlatformIDs(0, NULL, &platformCount);
	if (ret != CL_SUCCESS || !platformCount) {
		printf("Could not find any opencl platforms on your computer.\n");
		return;
	}
	printf("Found %u platform(s) on your computer.\n", platformCount);

	platformids = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);

	ret = clGetPlatformIDs(platformCount, platformids, NULL);
	if (ret != CL_SUCCESS) {
		printf("Error while fetching platform ids.\n");
		free(platformids);
		return;
	}

	int i,j; // Iterate through each platform and print its devices
	for (i = 0; i < platformCount; i++) {
		char str[80];
		// Print platform info.
		ret = clGetPlatformInfo(platformids[i], CL_PLATFORM_NAME, 80, str, NULL);
		if (ret != CL_SUCCESS) {
			printf("\tError while fetching platform info.\n");
			continue;
		}
		printf("Devices on platform %d, \"%s\":\n", i, str);
		ret = clGetDeviceIDs(platformids[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
		if (ret != CL_SUCCESS) {
			printf("\tError while fetching device ids.\n");
			continue;
		}
		if (!deviceCount) {
			printf("\tNo devices found for this platform.\n");
			continue;
		}
		deviceids = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);

		ret = clGetDeviceIDs(platformids[i], CL_DEVICE_TYPE_GPU, deviceCount, deviceids, NULL);
		if (ret != CL_SUCCESS) {
			printf("\tError while getting device ids.\n");
			free(deviceids);
			continue;
		}

		for (j = 0; j < deviceCount; j++) {
			// Print platform info.
			ret = clGetDeviceInfo(deviceids[j], CL_DEVICE_NAME, 80, str, NULL);
			if (ret != CL_SUCCESS) {
				printf("\tError while getting device info.\n");
				free(deviceids);
				continue;
			}
			printf("\tDevice %d: %s\n", j, str);
		}
		free(deviceids);
	}
	free(platformids);
}

// main reads the command line arguments and then starts the miner. The program
// will exit if there are any errors.
int main(int argc, char *argv[]) {
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_uint platformid = 0, deviceidx = 0;
	int i;
	unsigned cycles_per_iter;
	char hostname[128] = "localhost";
	char port_number[7] = ":9980";
	double hash_rate;

	// Parse args.
	cycles_per_iter = DEFAULT_CPI;
	for (i = 1; i < argc; i++) {
		char c = argv[i][1]; // If argv is "-c" then arv[i][1] is 'c'
		if (c == '-') {
			// If they did --flag, make c the next char.
			c = argv[i][2];
		}
		switch (c) {
		case 'h':
			printf("\nUsage:\n\n");
			printf("\t C - cycles per iter: Number of kernel executions between Sia API calls and hash rate updates\n");
			printf("\t\tA low C will cause instability. As a rule of thumb, the hashrate should only be updating a few times per second.\n");
			printf("\t\tDefault is %u.\n", DEFAULT_CPI);
			printf("\n");
			printf("\t I - intensity: This is the amount of work sent to the GPU in one batch.\n");
			printf("\t\tInterpretation is 2^intensity; the default is 16. Lower if GPU crashes or\n");
			printf("\t\tif more desktop interactivity is desired. Highest hashrate is typically at 22-25.\n");
			printf("\n");
			printf("\t H - host: which host name to use when talking to the siad api. (default: %s)\n", hostname);
			printf("\n");
			printf("\t P - port: which port to use when talking to the siad api. (e.g. -p :9980)\n");
			printf("\n");
			printf("\t p - OpenCL platform ID: Just what it says on the tin. If you're finding no GPUs,\n");
			printf("\t\tyet you're sure they exist, try a value other than 0, like 1, or 2. Default is 0.\n");
			printf("\n");
			printf("\t d - OpenCL device ID: Self-explanatory; it's the GPU index. Note that different\n");
			printf("\t\tOpenCL platforms will likely have different devices available. Default is 0.\n");
			printf("\n");
			printPlatformsAndDevices();
			printf("\n");
			exit(0);
			break;
		case 'I':
			if (++i >= argc) {
				printf("Please pass in a number following your flag (e.g. -I 22)\n");
				exit(1);
			}
			// atoi returns 0 on error.
			intensity = atoi(argv[i]);
			if (intensity == 0 && argv[i][0] != '0') { // Check if atoi returned 0 because of an error
				printf("Invalid number passed to \'-I\'\n");
				exit(1);
			}
			
			if(intensity < MIN_INTENSITY || intensity > MAX_INTENSITY) {
				printf("intensity must be between %u and %u. %u is invalid\n", MIN_INTENSITY, MAX_INTENSITY, intensity);
				printf("Note that the default intensity is %d\n", DEFAULT_INTENSITY);
				exit(1);
			}
			printf("Intensity set to %u\n", intensity);
			break;
		case 'p':
			if (++i >= argc) {
				printf("Please pass in a number following your flag (e.g. -p 1)\n");
				exit(1);
			}
			platformid = atoi(argv[i]);
			if (platformid == 0 && argv[i][0] != '0') {
				printf("Invalid number passed to \'-p\'\n");
				exit(1);
			}
			break;
		case 'd':
			if (++i >= argc) {
				printf("Please pass in a number following your flag (e.g. -d 1)\n");
				exit(1);
			}
			deviceidx = atoi(argv[i]);
			if (deviceidx == 0 && argv[i][0] != '0') {
				printf("Invalid number passed to \'-d\'\n");
				exit(1);
			}
			break;
		case 'C':
			if (++i >= argc) {
				printf("Please pass in a number following your flag (e.g. -C 10)\n");
				exit(1);
			}
			cycles_per_iter = atoi(argv[i]);
			if (cycles_per_iter == 0 && argv[i][0] != '0') {
				printf("Invalid number passed to \'-C\'\n");
				exit(1);
			}

			if(cycles_per_iter < MIN_CPI || cycles_per_iter > MAX_CPI) {
				printf("cycles per iter must be between %u and %u. %u is invalid\n", MIN_CPI, MAX_CPI, cycles_per_iter);
				printf("Note that the default cycles per iter is %d\n", DEFAULT_CPI);
				exit(1);
			}
			printf("Cycles per iteration set to %u\n", cycles_per_iter);
			break;
		case 'H':
			if (++i >= argc) {
				printf("Please pass in a host name following your flag (e.g. -H localhost)\n");
				exit(1);
			}
			strcpy(hostname, argv[i]);
			printf("Host name set to %s\n", hostname);
			break;
		case 'P':
			if (++i >= argc) {
				printf("Please pass in a port number following your flag (e.g. -P :9980)\n");
				exit(1);
			}
			if (strlen(argv[i]) < 6) {
				strcpy(port_number, argv[i]);
			} else {
				printf("Invalid port passed in as flag\n");
				exit(1);
			}
			printf("Port set to %s\n", port_number);
			break;
		default:
			printf("Please use a valid flag. Use \"--help\" for options\n");
			exit(1);
			break;
		}
	}

	// Set siad URL.
	set_host(hostname, port_number);

	// Load kernel source file.
	printf("Initializing...\n");
	fflush(stdout);
	FILE *fp;
	const char fileName[] = "./sia-gpu-miner.cl";
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
	
	selectOCLDevice(&platform_id, &device_id, platformid, deviceidx);
	
	// Make sure the device can handle our local item size.
	size_t max_group_size = 0;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_group_size, NULL);
	if (ret != CL_SUCCESS) { printf("failed to get Device IDs: %d\n", ret); exit(1); }
	if (local_item_size > max_group_size) {
		printf("Selected device cannot handle work groups larger than %zu.\n", local_item_size);
		printf("Using work groups of size %zu instead.\n", max_group_size);
		local_item_size = max_group_size;
	}

	// Create OpenCL Context.
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create command queue.
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create Buffer Objects.
	blockHeadermobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 80 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create blockHeadermobj buffer: %d\n", ret); exit(1); }
	nonceOutmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(uint8_t), NULL, &ret);
	if (ret != CL_SUCCESS) { printf("failed to create nonceOutmobj buffer: %d\n", ret); exit(1); }

	// Create kernel program from source file.
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if (ret != CL_SUCCESS) { printf("failed to crate program with source: %d\n", ret); exit(1); }
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		// Print information about why the build failed. This code is from
		// StackOverflow.
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

	// Create data parallel OpenCL kernel.
	kernel = clCreateKernel(program, "nonceGrind", &ret);

	// Set OpenCL kernel arguments.
	void *args[] = { &blockHeadermobj, &nonceOutmobj };
	for (i = 0; i < 2; i++) {
		ret = clSetKernelArg(kernel, i, sizeof(cl_mem), args[i]);
		if (ret != CL_SUCCESS) {
			printf("failed to set kernel arg %d (error code %d)\n", i, ret);
			exit(1);
		}
	}
	printf("\n");

	// Initialize network connection variables.
	init_network();

	// Grind nonces until SIGINT.
	signal(SIGINT, quitSignal);
	while (!quit) {
		// Repeat until no block is found.
		do {
			hash_rate = grindNonces(cycles_per_iter);
		} while (hash_rate == -1 && !quit);

		if (!quit) {
			printf("\rMining at %.3f MH/s\t%u blocks mined", hash_rate, blocks_mined);
			fflush(stdout);
		}
	}

	// Finalization.
	ret = clFlush(command_queue);   
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(blockHeadermobj);
	ret = clReleaseMemObject(nonceOutmobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);	

	free_network();
	free(source_str);
	return 0;
}
