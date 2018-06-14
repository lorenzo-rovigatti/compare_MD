/*
 * interaction.h
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#ifndef SRC_INTERACTION_H_
#define SRC_INTERACTION_H_

#include "defs.h"

int LJ_interaction(number r_sqr, number *energy, number *force_module_over_r);

#endif /* SRC_INTERACTION_H_ */
