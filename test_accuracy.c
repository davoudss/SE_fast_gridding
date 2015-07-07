/* Based on the code by Dag Lindbo,
 * INPUT:
 * x:	coordinates
 * q:	charges
 * N:	number of atoms
 * opt: constants to pass
 * 
 * OUTPUT: 
 * H: 	Interpolated values using a FGG to the grids.
 *
 * Davoud Saffar Shamshirgar, 2013,05,28
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "SE_fgg.h"
#include "SE_general.h"
#include "SE_direct.h"
#include <string.h>

#ifndef VERBOSE
#define VERBOSE 0
#endif


double norm(double *, double *, int);


int
spectral_ewald(double *x, double *q, SE_opt opt,
	       double xi, double* force_phi, double *phi_energy);

int main(int argc, char* argv[])
{
  const int N = atoi(argv[1]);
  int P = atoi(argv[2]);
 
  int M;
  double m, L=3.1814, xi=8.00, eps=1;

  SE_FGG_params params = {.N = N};
  SE_state st;
  
  // Initialize the system
  SE_init_unit_system(&st, &params);

  // Initialize parameters for SE
  SE_init_params(&M,&P,&m,eps,L,xi,st.q,N);

  SE_opt opt = {.m = m, .box = {L, L, L}, .P = P, .M = M, .N=N};
  
  // to run direct method
  ewald_opts ED_opt = {.xi=xi, .box = {L, L, L}};

  ED_opt.layers = (int) (M-1)/2;


#ifdef POTENTIAL
//printf("\n***************** potential *****************\n");
 
 // allocate output array
  double *phi_direct = malloc(sizeof(double)*N);
 
  SE_fp_set_zero(phi_direct,N);

  SE3P_direct_fd(phi_direct, st.x, st.q, N, ED_opt);
  spectral_ewald(st.x,st.q,opt,xi,st.phi,st.phi);

  printf("error in potential: %g \n",norm(phi_direct,st.phi,N));


#endif


#ifdef FORCE
//printf("\n***************** force *****************\n");
 
 // allocate output array
  double *force        = malloc(sizeof(double)*3*N);
  double *force_direct = malloc(sizeof(double)*3*N);
  
  SE_fp_set_zero(force,N);
  SE_fp_set_zero(force_direct,N);

  // to run direct method
  SE3P_direct_force_fd(force_direct, st.x, st.q, N, ED_opt);
  spectral_ewald(st.x,st.q,opt,xi,force,st.phi);
 
  printf("error in force: %g \n",norm(force,force_direct,3*N));


#endif

#ifdef POTENTIAL
  free(phi_direct);
#endif
#ifdef FORCE
  free(force);free(force_direct);
#endif

  return 0;

}

double norm(double *a, double *b, int N)
{
  int i;
  double s=0,tmp=0;

  for (i=0;i<N;i++)
    {
      tmp=(a[i]-b[i]);
      s+=tmp*tmp;
    }
  return sqrt(s/N);
}
