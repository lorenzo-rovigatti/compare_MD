/*
 * interaction.cpp
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#include "interaction.h"

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
