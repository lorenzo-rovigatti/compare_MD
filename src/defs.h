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

#define NO_OVERLAP 0
#define OVERLAP 1

#define RCUT 2.5
#define OVERLAP_THRESHOLD 0.9
#define TEMPERATURE 1.0

typedef double number;
typedef number vector[3];

#endif /* SRC_DEFS_H_ */
