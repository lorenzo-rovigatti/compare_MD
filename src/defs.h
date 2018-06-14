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

#define RCUT 2.5
#define OVERLAP_THRESHOLD 0.95
#define TEMPERATURE 1.0

typedef double number;

typedef struct {
	number x, y, z;
} vector;

typedef struct {
	int N;
	number box_side;
	vector *positions;
	vector *velocities;
	vector *forces;
} MD_system;

#endif /* SRC_DEFS_H_ */
