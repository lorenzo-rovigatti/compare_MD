/*
 * utils.cpp
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#include "utils.h"

#include <cstdlib>
#include <cmath>

number gaussian() {
	static unsigned int isNextG = 0;
	static double nextG;
	double toRet;
	double u, v, w;

	if(isNextG) {
		isNextG = 0;
		return nextG;
	}

	w = 2.;
	while(w >= 1.0) {
		u = 2. * drand48() - 1.0;
		v = 2. * drand48() - 1.0;
		w = u * u + v * v;
	}

	w = sqrt((-2. * log(w)) / w);
	toRet = u * w;
	nextG = v * w;
	isNextG = 1;

	return toRet;
}
