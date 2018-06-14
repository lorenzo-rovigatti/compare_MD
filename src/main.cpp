/*
 * main.c
 *
 *  Created on: 13 giu 2018
 *      Author: lorenzo
 */

#include "defs.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>

int LJ_interaction(number r_sqr, number *energy, number *force_module_over_r) {
	if(r_sqr > SQR(RCUT)) {
		*energy = *force_module_over_r = 0.;
		return NO_OVERLAP;
	}
	if(r_sqr < SQR(OVERLAP_THRESHOLD)) {
		*energy = *force_module_over_r = 0.;
		return OVERLAP;
	}

	number lj_part = 1. / CUB(r_sqr);
	*energy = 4 * (SQR(lj_part) - lj_part);
	*force_module_over_r = -4 * (lj_part - 2 * SQR(lj_part)) / r_sqr;

	return NO_OVERLAP;
}

void init_configuration(int N, number box_side, vector *positions) {
	int i = 0;
	for(i = 0; i < N; i++) {

	}
}

int main(int argc, char *argv[]) {
	if(argc < 3) {
		fprintf(stderr, "Usage is %s N density [sim_type]\n", argv[0]);
		exit(1);
	}

	srand48(13245124);

	int N = atoi(argv[1]);
	number density = atof(argv[2]);
	number box_side = pow(N / density, 1./3.);

	vector *positions = (vector *) malloc(sizeof(vector) * N);
	vector *velocities = (vector *) malloc(sizeof(vector) * N);
	vector *forces = (vector *) malloc(sizeof(vector) * N);

	init_configuration(N, box_side, positions);

	free(forces);
	free(velocities);
	free(positions);

	return 0;
}
