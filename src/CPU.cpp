/*
 * CPU.cpp
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#include "CPU.h"
#include "utils.h"

#include <cstdlib>
#include <cmath>

int LJ_interaction(number r_sqr, number *energy, number *force_module_over_r) {
	int is_overlap = (r_sqr < SQR(OVERLAP_THRESHOLD));

	if(r_sqr > SQR(RCUT)) {
		*energy = *force_module_over_r = 0.;
		return is_overlap;
	}

	number lj_part = 1. / CUB(r_sqr);
	*energy = 4 * (SQR(lj_part) - lj_part);
	*force_module_over_r = 24 * (lj_part - 2 * SQR(lj_part)) / r_sqr;

	return is_overlap;
}

void CPU_first_step(MD_system *syst) {
	int i = 0;
	for(i = 0; i < syst->N; i++) {
		syst->velocities[i].x += syst->forces[i].x * DT / 2.;
		syst->velocities[i].y += syst->forces[i].y * DT / 2.;
		syst->velocities[i].z += syst->forces[i].z * DT / 2.;

		syst->positions[i].x += syst->velocities[i].x * DT;
		syst->positions[i].y += syst->velocities[i].y * DT;
		syst->positions[i].z += syst->velocities[i].z * DT;

		// here we also reset the forces
		syst->forces[i].x = syst->forces[i].y = syst->forces[i].z = 0.;
	}
}

void CPU_force_calculation(MD_system *syst) {
	syst->U = 0.;
	// N^2 calculations, very inefficient!
	int i, j;
	for(i = 0; i < syst->N; i++) {
		vector *p_position = syst->positions + i;
		vector *p_force = syst->forces + i;
		// here we loop on particles with j > 1 since we can exploit Newton's third law
		for(j = i + 1; j < syst->N; j++) {
			vector *q_position = syst->positions + j;
			vector *q_force = syst->forces + j;

			vector dist = {
					q_position->x - p_position->x,
					q_position->y - p_position->y,
					q_position->z - p_position->z
			};
			dist.x -= syst->box_side * rint(dist.x / syst->box_side);
			dist.y -= syst->box_side * rint(dist.y / syst->box_side);
			dist.z -= syst->box_side * rint(dist.z / syst->box_side);

			number r_sqr = DOT(dist, dist);
			number energy, force_module_over_r;
			LJ_interaction(r_sqr, &energy, &force_module_over_r);
			syst->U += energy;

			p_force->x += dist.x * force_module_over_r;
			p_force->y += dist.y * force_module_over_r;
			p_force->z += dist.z * force_module_over_r;

			q_force->x -= dist.x * force_module_over_r;
			q_force->y -= dist.y * force_module_over_r;
			q_force->z -= dist.z * force_module_over_r;
		}
	}
}

void CPU_thermalise(MD_system *syst) {
	number rescale_factor = sqrt(TEMPERATURE);
	int i = 0;
	for(i = 0; i < syst->N; i++) {
		if(drand48() < THERMOSTAT_PT) {
			syst->velocities[i].x = rescale_factor * gaussian();
			syst->velocities[i].y = rescale_factor * gaussian();
			syst->velocities[i].z = rescale_factor * gaussian();
		}
	}
}

void CPU_second_step(MD_system *syst) {
	syst->K = 0.;
	int i = 0;
	for(i = 0; i < syst->N; i++) {
		syst->velocities[i].x += syst->forces[i].x * DT / 2.;
		syst->velocities[i].y += syst->forces[i].y * DT / 2.;
		syst->velocities[i].z += syst->forces[i].z * DT / 2.;

		syst->K += 0.5 * DOT(syst->velocities[i], syst->velocities[i]);
	}
}
