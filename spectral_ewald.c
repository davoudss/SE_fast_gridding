/* Spectral Ewald to calcualte force code, based on the spectral_ewald
 * code by Dag Lindbo.
 * Davoud Saffar Shamshirgar, 2013,05,28
 */

#include <stdio.h>
#include <stdlib.h>

#include "SE_fgg.h"
#include <assert.h>
#include "SE_general.h"

#ifdef FGG_SPLIT
#define PRECOMP_FGG_EXPA 1
#else
#define PRECOMP_FGG_EXPA 0
#endif

#define Z(i,j,k) Z[M*(i*M+j)+k]
int
spectral_ewald(double *x, double *q, SE_opt opt,
	       double xi, double* force_phi, double *phi_energy)
{
  // parameters and constants
  int M,N,P;
  double m,w,eta,c;
  double start = SE_gettime();
 
  // creating and set new parameters
  parse_params(&opt,xi);

  // setting parameters for use
  m = opt.m; w = opt.w; eta = opt.eta; c = opt.c; 
  M = opt.M; N = opt.N; P = opt.P;
  int M3 = M*M*M;

  if(VERBOSE){
    printf("[ SE ] Grid: [%d %d %d]\tGaussian: {P = %d, w=%.2f, m=%f}\n", M, M, M, P, w, m);
    printf("[ SE ] {eta=%.4f, c=%.2f}\n", eta, c);
  }

  double *H_in = malloc(sizeof(double)*M3);
  t_complex *H_out = malloc(sizeof(t_complex)*M3); // to copy real to comp for fft

  // to grid function **************************************
  // pack parameters
  SE_FGG_params params;
  SE_FGG_FCN_params(&params, &opt, N);

  // scratch arrays
  SE_FGG_work work;
  
  if(PRECOMP_FGG_EXPA){
#ifdef POTENTIAL
     SE_FGG_allocate_workspace(&work, &params,true,true);
#endif
#ifdef FORCE
     SE_FGG_allocate_workspace_SSE_force(&work, &params,true,true);
#endif
  }
  else
    SE_FGG_allocate_workspace(&work, &params,true,false);

  // initialize the output with zeros
  SE_fp_set_zero(H_in,SE_prod3(params.dims));

  // coordinates and charges
#ifdef POTENTIAL
  SE_state st = {.x = x, .q = q};
#else
  SE_state st = {.x = x, .q = q, .phi = phi_energy};
#endif

  if(VERBOSE)
	printf("[SE FG(G)] N=%d, P=%d\n",N,params.P);
  
    int flag;
    double scalar;
    double *Z = SE_FGG_MALLOC(sizeof(double)*M3);
    fftw_plan p1=NULL,p2=NULL;

   // now do the work
   SE_FGG_base_gaussian(&work, &params);
   if(VERBOSE)
	printf("[SE FG(G) base ] N=%d, P=%d\n",N,params.P);

   if(PRECOMP_FGG_EXPA){
#ifdef POTENTIAL
     SE_FGG_expand_all(&work, &st, &params);
#ifdef SSE
     SE_FGG_grid_split_SSE_dispatch(&work, &st, &params);
#elif AVX
     SE_FGG_grid_split_AVX_dispatch(&work, &st, &params);
#endif
#endif

#ifdef FORCE
     SE_FGG_expand_all_SSE_force(&work, &st, &params);
#ifdef SSE
     SE_FGG_grid_split_SSE_dispatch_force(&work, &st, &params);
#elif AVX
     SE_FGG_grid_split_AVX_dispatch_force(&work, &st, &params);
#endif
#endif
   }
   else
     SE_FGG_grid(&work, &st, &params);
   
   if(VERBOSE)
	printf("[SE FG(G) GRID] N=%d, P=%d\n",N,params.P);

   SE_FGG_wrap_fcn(H_in, &work, &params);


  //*************************** END griding

  
  // transform and shift
  copy_r2c(H_in,H_out,M3);
  p1=fftw_plan_dft_3d(M,M,M, H_out, H_out, FFTW_FORWARD, FFTW_ESTIMATE);
  p2=fftw_plan_dft_3d(M,M,M, H_out, H_out, FFTW_BACKWARD, FFTW_ESTIMATE);

  fftw_execute(p1);

  // scale
  scalar = -(1.-eta)/(4.*xi*xi);
  scaling(scalar,Z,M,M,M,opt.box);
  Z(0,0,0) = 0.;
  
  flag = 1;  // to multiply. flag = -1 for division!
  product_rc(H_out,Z,H_out,M,M,M,flag);

  fftw_execute(p2);
  copy_c2r(H_out,H_in,M*M*M);

  // spread and integrate
#ifdef POTENTIAL
   SE_FGG_extend_fcn(&work, H_in, &params);

   // Integration
   if(PRECOMP_FGG_EXPA)
#ifdef SSE
     SE_FGG_int_split_SSE_dispatch(force_phi, &work, &params);
#elif AVX
     SE_FGG_int_split_AVX_dispatch(force_phi, &work, &params);
#endif
   else
     SE_FGG_int(force_phi, &work, &st, &params);
#endif

#ifdef FORCE
  SE_FGG_extend_fcn(&work, H_in, &params);

  //Integration
  if(PRECOMP_FGG_EXPA){
#ifdef SSE
    SE_FGG_int_split_SSE_dispatch_force(force_phi, &st,&work, &params);
#elif AVX
    SE_FGG_int_split_AVX_dispatch_force(force_phi, &st,&work, &params);
#endif
  }
  else{
    SE_FGG_int_force(force_phi, &work, &st, &params);
  }
#endif

    if(VERBOSE)
	printf("[SE FG(G) INT] N=%d, P=%d\n",N,params.P);

  double stop=SE_gettime();
#ifdef SSE
  printf("Duration of SSE: %f \n",stop-start);
#elif AVX
  printf("Duration of AVX: %f \n",stop-start);
#endif


  fftw_destroy_plan(p1); 
  fftw_destroy_plan(p2); 
	
  return 0;
}
