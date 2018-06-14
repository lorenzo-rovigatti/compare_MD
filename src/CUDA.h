/*
 * CUDA.h
 *
 *  Created on: 14 giu 2018
 *      Author: lorenzo
 */

#ifndef SRC_CUDA_H_
#define SRC_CUDA_H_

#include "defs.h"

void CUDA_first_step(MD_system *syst);
void CUDA_force_calculation(MD_system *syst);
void CUDA_thermalise(MD_system *syst);
void CUDA_second_step(MD_system *syst);

#endif /* SRC_CUDA_H_ */
