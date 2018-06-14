/*
 * CPU.h
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#ifndef SRC_CPU_H_
#define SRC_CPU_H_

#include "defs.h"

int LJ_interaction(number r_sqr, number *energy, number *force_module_over_r);
void CPU_first_step(MD_system *syst);
void CPU_force_calculation(MD_system *syst);
void CPU_thermalise(MD_system *syst);
void CPU_second_step(MD_system *syst);

#endif /* SRC_CPU_H_ */
