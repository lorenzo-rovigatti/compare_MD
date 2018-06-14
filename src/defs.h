/*
 * defs.h
 *
 *  Created on: 13 giu 2018
 *      Author: lorenzo
 */

#ifndef SRC_DEFS_H_
#define SRC_DEFS_H_

#define SQR(x) ((x) * (x))
#define CUB(x) ((x) * (x) * (x))
#define DOT(v, w) ((v).x * (w).x + (v).y * (w).y + (v).z * (w).z)

#define NO_OVERLAP 0
#define OVERLAP 1

// Simulation parameters
#define RCUT 2.5
#define OVERLAP_THRESHOLD 0.9
#define TEMPERATURE 1.0
#define DT 0.001
#define THERMOSTAT_PT 0.1
#define THERMOSTAT_EVERY 537
#define PRINT_ENERGY_EVERY 100

typedef double number;

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>

typedef float4 vector;
#else
typedef struct {
	number x, y, z;
} vector;
#endif

typedef struct {
	int N;
	number box_side;
	number U;
	number K;
	vector *positions;
	vector *velocities;
	vector *forces;
#ifdef CUDA
	vector *d_positions;
	vector *d_velocities;
	vector *d_forces;
#endif
} MD_system;

#endif /* SRC_DEFS_H_ */
