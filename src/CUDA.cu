/*
 * CUDA.cpp
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#include "CUDA.h"
#include "CUDA_device_utils.h"
#include "CUDA_kernels.cuh"

#include <cstdlib>
#include <cstdio>

#define CHECK_CUDA_ERROR(msg) __check_CUDA_error(msg, __FILE__, __LINE__)
inline void __check_CUDA_error(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();

	if(cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int) err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void CUDA_init(MD_system *syst) {
	cudaDeviceProp props = set_device_automatically();
	syst->kernel_threads_per_block = 2 * props.warpSize;

	syst->kernel_blocks.x = syst->N / syst->kernel_threads_per_block + ((syst->N % syst->kernel_threads_per_block == 0) ? 0 : 1);
	if(syst->kernel_blocks.x == 0) syst->kernel_blocks.x = 1;
	syst->kernel_blocks.y = syst->kernel_blocks.z = 1;

	fprintf(stderr, "CUDA device: %s\n", props.name);
	fprintf(stderr, "CUDA threads per block: %d\n", syst->kernel_threads_per_block);
	fprintf(stderr, "CUDA blocks: (%d, %d, %d)\n", syst->kernel_blocks.x, syst->kernel_blocks.y, syst->kernel_blocks.z);

	syst->vector_size = sizeof(vector) * syst->N;
	cudaMalloc(&syst->d_positions, syst->vector_size);
	cudaMalloc(&syst->d_velocities, syst->vector_size);
	cudaMalloc(&syst->d_forces, syst->vector_size);

	cudaMemcpyToSymbol(MD_N, &syst->N, sizeof(int));
	cudaMemcpyToSymbol(MD_box_side, &syst->box_side, sizeof(number));

	CPU_to_CUDA(syst);
}

void CUDA_clean(MD_system *syst) {
	cudaFree(&syst->d_positions);
	cudaFree(&syst->d_velocities);
	cudaFree(&syst->d_forces);
}

void CUDA_to_CPU(MD_system *syst) {
	cudaMemcpy(syst->positions, syst->d_positions, syst->vector_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(syst->velocities, syst->d_velocities, syst->vector_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(syst->forces, syst->d_forces, syst->vector_size, cudaMemcpyDeviceToHost);
}

void CPU_to_CUDA(MD_system *syst) {
	cudaMemcpy(syst->d_positions, syst->positions, syst->vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(syst->d_velocities, syst->velocities, syst->vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(syst->d_forces, syst->forces, syst->vector_size, cudaMemcpyHostToDevice);
}

void CUDA_first_step(MD_system *syst) {
	first_step_kernel
		<<<syst->kernel_blocks, syst->kernel_threads_per_block>>>
		(syst->d_positions, syst->d_velocities, syst->d_forces);
		CHECK_CUDA_ERROR("first_step error");
}

void CUDA_force_calculation(MD_system *syst) {
	force_calculation_kernel
		<<<syst->kernel_blocks, syst->kernel_threads_per_block>>>
		(syst->d_positions, syst->d_forces);
		CHECK_CUDA_ERROR("force_calculation error");
}

void CUDA_thermalise(MD_system *syst) {

}

void CUDA_second_step(MD_system *syst) {
	second_step_kernel
		<<<syst->kernel_blocks, syst->kernel_threads_per_block>>>
		(syst->d_velocities, syst->d_forces);
		CHECK_CUDA_ERROR("second_step error");
}
