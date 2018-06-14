/*
 * main.c
 *
 *  Created on: 13 giu 2018
 *      Author: lorenzo
 */

#include "defs.h"
#include "CPU.h"
#include "utils.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>

void init_configuration(MD_system *syst) {
	// initialise the positions
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

	// extract the velocities from a Maxwell-Boltzmann
	number rescale_factor = sqrt(TEMPERATURE);
	for(i = 0; i < syst->N; i++) {
		syst->velocities[i].x = rescale_factor * gaussian();
		syst->velocities[i].y = rescale_factor * gaussian();
		syst->velocities[i].z = rescale_factor * gaussian();
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
		fprintf(stderr, "Usage is %s N density [steps=100000]\n", argv[0]);
		exit(1);
	}

	srand48(13245124);

	MD_system syst;

	syst.N = atoi(argv[1]);
	number density = atof(argv[2]);
	syst.box_side = pow(syst.N / density, 1./3.);
	int steps = 100000;
	if(argc > 3) {
		steps = atoi(argv[3]);
	}

	syst.positions = (vector *) calloc(syst.N, sizeof(vector));
	syst.velocities = (vector *) calloc(syst.N, sizeof(vector));
	syst.forces = (vector *) calloc(syst.N, sizeof(vector));

	init_configuration(&syst);
	// the explicit cast is there to remove a warning issued by g++
	print_cogli1_configuration(&syst, (char *)"initial.mgl");

	int step;
	for(step = 0; step < steps; step++) {
		CPU_first_step(&syst);
		CPU_force_calculation(&syst);
		CPU_second_step(&syst);

		if(step % THERMOSTAT_EVERY == 0) {
			CPU_thermalise(&syst);
		}

		if(step % PRINT_ENERGY_EVERY == 0) {
			fprintf(stdout, "%d %lf %lf %lf\n", step, syst.U / syst.N, syst.K/ syst.N, (syst.U + syst.K) / syst.N);
		}
	}

	print_cogli1_configuration(&syst, (char *)"last.mgl");

	free(syst.forces);
	free(syst.velocities);
	free(syst.positions);

	return 0;
}
