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

void init_configuration(MD_system *syst) {
	int i = 0;
	for(i = 0; i < syst->N; i++) {
		int done = 0;
		vector trial_position;
		while(!done) {
			trial_position.x = drand48() * syst->box_side;
			trial_position.y = drand48() * syst->box_side;
			trial_position.z = drand48() * syst->box_side;
			// N^2 check to look for "overlaps"
			int j = 0;
			done = 1;
			for(j = 0; j < i && done; j++) {
				vector *other_position = syst->positions + j;
				vector dist = {
						trial_position.x - other_position->x,
						trial_position.y - other_position->y,
						trial_position.z - other_position->z
				};
				dist.x -= syst->box_side * rint(dist.x / syst->box_side);
				dist.y -= syst->box_side * rint(dist.y / syst->box_side);
				dist.z -= syst->box_side * rint(dist.z / syst->box_side);

				number r_sqr = DOT(dist, dist);
				number energy, force;
				if(LJ_interaction(r_sqr, &energy, &force) == OVERLAP) {
					done = 0;
				}
			}
		}

		syst->positions[i] = trial_position;

		if(syst->N > 10 && i % (syst->N/10) == 0) {
			fprintf(stderr, "Inserted %d%% of the particles (%d/%d)\n", i*100/syst->N, i, syst->N);
		}
	}
}

void print_cogli1_configuration(MD_system *syst, char *filename) {
	FILE *out = fopen(filename, "w");
	if(out == NULL) {
		fprintf(stderr, "File '%s' is not writable", filename);
		exit(1);
	}

	fprintf(out, ".Box:%lf,%lf,%lf\n", syst->box_side, syst->box_side, syst->box_side);
	int i;
	for(i = 0; i < syst->N; i++) {
		fprintf(out, "%lf %lf %lf @ 0.5 C[blue]\n", syst->positions[i].x, syst->positions[i].y, syst->positions[i].z);
	}

	fclose(out);
}

int main(int argc, char *argv[]) {
	if(argc < 3) {
		fprintf(stderr, "Usage is %s N density [sim_type]\n", argv[0]);
		exit(1);
	}

	srand48(13245124);

	MD_system syst;

	syst.N = atoi(argv[1]);
	number density = atof(argv[2]);
	syst.box_side = pow(syst.N / density, 1./3.);

	syst.positions = (vector *) malloc(sizeof(vector) * syst.N);
	syst.velocities = (vector *) malloc(sizeof(vector) * syst.N);
	syst.forces = (vector *) malloc(sizeof(vector) * syst.N);

	init_configuration(&syst);
	// the explicit cast is there to remove a warning issued by g++
	print_cogli1_configuration(&syst, (char *)"initial.mgl");

	free(syst.forces);
	free(syst.velocities);
	free(syst.positions);

	return 0;
}
