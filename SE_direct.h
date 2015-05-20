#include "SE_fgg.h"
#ifndef SE_DIRECT_H
#define SE_DIRECT_H

//#define PI 3.141592653589793

#define __MALLOC malloc
#define __PRINTF printf
#define __FREE free

typedef struct 
{
    double box[3];
    double xi;
    int layers; 
	double rc;
} ewald_opts;


void SE3P_direct(double*, const double *, const double*, int, 
		 const ewald_opts);

void SE3P_direct_force(double*, const double *, const double*, int, const ewald_opts);

void SE3P_direct_real(double*, const double* , const double* , int , const ewald_opts);

void SE3P_direct_self(double* , const double* , int , const ewald_opts);

#endif
