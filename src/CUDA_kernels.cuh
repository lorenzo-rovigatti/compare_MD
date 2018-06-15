#include "defs.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>

/// threads per block
#define TINBLOCK (blockDim.x*blockDim.y)
/// thread id relative to its block
#define TID (blockDim.x*threadIdx.y + threadIdx.x)
/// block id
#define BID (gridDim.x*blockIdx.y + blockIdx.x)
/// thread id
#define IND (TINBLOCK * BID + TID)

__constant__ int MD_N[1];
__constant__ number MD_box_side[1];

__global__ void first_step_kernel(vector *positions, vector *velocities, vector *forces) {
	if(IND >= MD_N[0]) return;

	const vector F = forces[IND];

	vector r = positions[IND];
	vector v = velocities[IND];

	v.x += F.x * (DT * (number) 0.5f);
	v.y += F.y * (DT * (number) 0.5f);
	v.z += F.z * (DT * (number) 0.5f);

	r.x += v.x * DT;
	r.y += v.y * DT;
	r.z += v.z * DT;

	velocities[IND] = v;
	positions[IND] = r;
}

__global__ void force_calculation_kernel(vector *positions, vector *forces) {
	if(IND >= MD_N[0]) return;

	vector F = {
			0.f,
			0.f,
			0.f,
			0.f
	};
	vector p_position = positions[IND];

	for(int j = 0; j < MD_N[0]; j++) {
		if(j != IND) {
			vector q_position = positions[j];

			vector dist = {
					q_position.x - p_position.x,
					q_position.y - p_position.y,
					q_position.z - p_position.z,
					0.f
			};
			dist.x -= MD_box_side[0] * rint(dist.x / MD_box_side[0]);
			dist.y -= MD_box_side[0] * rint(dist.y / MD_box_side[0]);
			dist.z -= MD_box_side[0] * rint(dist.z / MD_box_side[0]);

			number r_sqr = DOT(dist, dist);
			number lj_part = 1.f / CUB(r_sqr);

			number force_module_over_r = 24.f * (lj_part - 2.f * SQR(lj_part)) / r_sqr;

			F.x += force_module_over_r * dist.x;
			F.y += force_module_over_r * dist.y;
			F.z += force_module_over_r * dist.z;
			F.w += 4.f * (SQR(lj_part) - lj_part);
		}
	}

	forces[IND] = F;
}

__global__ void second_step_kernel(vector *velocities, vector *forces) {
	if(IND >= MD_N[0]) return;

	const vector F = forces[IND];
	vector v = velocities[IND];

	v.x += F.x * (DT * (number) 0.5f);
	v.y += F.y * (DT * (number) 0.5f);
	v.z += F.z * (DT * (number) 0.5f);
	v.w = 0.5f * DOT(v, v);

	velocities[IND] = v;
}
