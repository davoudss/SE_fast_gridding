#ifndef __MATHINT_H_
#define __MATHINT_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#if (defined __AVX__ || defined __SSE4_2__)
#include "math_x86.h"
#endif


#define pi     3.14159265358979323846264338327950288
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a > b) ? b : a)
#define EPS    1.110223024625157e-16
#define EULER  0.577215664901532860606512090082402431042

void glwt_fast(int, double, double, double*, double*);
void glwt(int, double, double, double*, double*);
void trapz_wt(int, double*, double*);
double expint(int, double);
double matlab_expint(double);
double IncompBesselK0_int_inf(double a, double b, int n);
double IncompBesselK0_int(double a, double b, int n, double* x, double* w);

#endif
