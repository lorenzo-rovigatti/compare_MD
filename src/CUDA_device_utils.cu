#include "CUDA_device_utils.h"

#include <cstdio>
#include <cstdlib>

int get_device_count() {
	int deviceCount = 0;
	if(cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount FAILED, CUDA Driver and Runtime CUDA Driver and Runtime version may be mismatched, exiting.\n");
		exit(-1);
	}

	return deviceCount;
}

void check_device_existance(int device) {
	if(device >= get_device_count()) {
		fprintf(stderr, "The selected device doesn't exist, exiting.\n");
		exit(-1);
	}
}

cudaDeviceProp get_current_device_prop() {
	int curr_dev;
	cudaGetDevice(&curr_dev);
	return get_device_prop(curr_dev);
}

cudaDeviceProp get_device_prop(int device) {
	check_device_existance(device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	return deviceProp;
}

cudaError_t set_device(int device) {
	check_device_existance(device);
	cudaThreadExit();
	return cudaSetDevice(device);
}

cudaDeviceProp set_device_automatically() {
	int trydev = 0;
	int ndev = -1;
	cudaGetDeviceCount(&ndev);
	fprintf(stderr, "The computer has %i devices\n", ndev);
	while(trydev < ndev) {
		fprintf(stderr, " - Trying device %i\n", trydev);
		cudaDeviceProp tryprop = get_device_prop(trydev);
		fprintf(stderr, " -- Device %i has properties %i.%i\n", trydev, tryprop.major, tryprop.minor);
		// we don't support old devices
		if (tryprop.major < 2 && tryprop.minor <= 2) {
			fprintf(stderr, " -- Device properties are not good. Skipping it\n");
			trydev ++;
			continue;
		}
		set_device (trydev);
		int *dummyptr = NULL;
		cudaError_t test = cudaMalloc((void **)&dummyptr, (size_t)sizeof(int));
		if(test == cudaSuccess) {
			fprintf(stderr, " -- Using device %i\n", trydev);
			cudaFree(dummyptr);
			break;
		}
		else {
			fprintf(stderr, " -- Device %i not available ...\n", trydev);
		}
		trydev++;
	}

	if(trydev == ndev) {
		fprintf(stderr, "No suitable devices available\n");
		exit(1);
	}

	fprintf(stderr, " --- Running on device %i\n", trydev);
	set_device(trydev);
	return get_device_prop(trydev);
}
