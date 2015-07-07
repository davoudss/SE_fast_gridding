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

#ifdef ONE_PERIODIC
void SE1P_direct_real(double*, const double*, const double*, int, const ewald_opts);

void SE1P_direct_real_rc(double*, const double*, const double*, int, const ewald_opts);

void SE1P_direct_fd(double*, const double *, const double*, int, const ewald_opts);

void SE1P_direct_k0(double*, const double *, const double*, int, const ewald_opts);

void SE1P_direct_self(double*, const double*, int, const ewald_opts);

#elif defined ( TWO_PERIODIC )
void SE2P_direct_real(double*, const double*, const double*, int, const ewald_opts);

void SE2P_direct_real_rc(double*, const double*, const double*, int, const ewald_opts);

void SE2P_direct_fd(double*, const double *, const double*, int, const ewald_opts);

void SE2P_direct_k0(double*, const double *, const double*, int, const ewald_opts);

void SE2P_direct_self(double*, const double*, int, const ewald_opts);

#else 
void SE3P_direct_real_rc(double*, const double* , const double* , int, const ewald_opts);

void SE3P_direct_fd(double*, const double *, const double*, int, const ewald_opts);

void SE3P_direct_force_fd(double*, const double *, const double*, int, const ewald_opts);

void SE3P_direct_self(double* , const double* , int , const ewald_opts);
#endif //PERIODICITY
#endif //SE_DIRECT_H
