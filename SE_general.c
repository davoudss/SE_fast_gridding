/* Extra modules to calculate error bounds and
 * other parameters
 * Davoud Saffar Shamshirgar, 2013,05,28
 */

#include "SE_general.h"
#include "SE_fgg.h"
#include <string.h>
#include <emmintrin.h>


// SE General Utility routines =====================================================

// -----------------------------------------------------------------------------
// unpacking params
void 
parse_params(SE_opt* opt, double xi)
{
  double      h0 = opt->box[0]/ (double) opt->M;
  double      w0 = (double) opt->P*h0/2.;
  double    eta0 = (2.0*w0*xi/opt->m)*(2.0*w0*xi/opt->m);
  double      c0 = 2.0*xi*xi/eta0;

  opt->h = h0;
  opt->c = c0;
  opt->w = w0;
  opt->eta = eta0;
  opt->xi = xi;
} 


// -----------------------------------------------------------------------------
// create k_space evctors
void
k_vec(int M, double* box,double *k1,double *k2, double *k3)
{
  int i,MM;
  double factor1 = 2.0*PI/box[0];
  double factor2 = 2.0*PI/box[1];
  double factor3 = 2.0*PI/box[2];
  double iter = 0.;
  

  if((M%2)==0){
    MM = M/2;
    for (i=0;i<=MM-1;i++)
      {
	k1[i] = factor1*iter;
	k2[i] = factor2*iter;
	k3[i] = factor3*iter;
	iter += 1.;
      }
		
    iter = (double) 0.-MM;
    for (i=0;i<=MM-1;i++)
      {
	k1[MM+i] = factor1*iter;
	k2[MM+i] = factor2*iter;
	k3[MM+i] = factor3*iter;
	iter += 1.;
      }
  }
  else {
    MM = (M-1)/2;
    iter = 0.;
    for (i=0;i<=MM;i++)
      {
	k1[i] = factor1*iter;
	k2[i] = factor2*iter;
	k3[i] = factor3*iter;
	iter += 1.;
      }
		
    iter = -MM;
    for (i=0;i<=MM-1;i++)
      {
	k1[MM+1+i] = factor1*iter;
	k2[MM+1+i] = factor2*iter;
	k3[MM+1+i] = factor3*iter;
	iter += 1.;
      }
  }

}

#if defined SSE || defined AVX
void scaling(double scalar, double *Z,
                 int n1, int n2, int n3, double* box)
{
  
  double *k1 = SE_FGG_MALLOC(sizeof(double)*n1);
  double *k2 = SE_FGG_MALLOC(sizeof(double)*n2);
  double *k3 = SE_FGG_MALLOC(sizeof(double)*n3);

  k_vec(n1,box,k1,k2,k3);

  int i,j,k,indij;
  double c = 4.*PI;
  double K2d[4]  MEM_ALIGNED;
  double K2s[4]  MEM_ALIGNED;
  __m128d rKi,rKij,rK2r,rK2e,rK2sr,rK2se,rK2dr,rK2de,rk3r,rk3e;
  __m128d rd,rscalar;

  rd   = _mm_div_pd(_mm_set1_pd( c ), _mm_set1_pd( (double) n1*n1*n1) );
  rscalar = _mm_set1_pd(scalar);

  for(i=0;i<n1;i++){
    rKi  = _mm_set1_pd(k1[i]*k1[i]);
    for(j=0;j<n2;j++){
      rKij = _mm_add_pd(rKi,_mm_set1_pd(k2[j]*k2[j]));
      indij = i*n3*n2+j*n2;
      for(k=0;k<n3;k+=4){
	rk3r = _mm_load_pd(k3 + k);
	rk3e = _mm_load_pd(k3 + k + 2);

	rK2r = _mm_add_pd(rKij,_mm_mul_pd(rk3r,rk3r));
	rK2e = _mm_add_pd(rKij,_mm_mul_pd(rk3e,rk3e));

	rK2dr= _mm_div_pd(rd,rK2r);
	rK2de= _mm_div_pd(rd,rK2e);

	rK2sr= _mm_mul_pd(rK2r,rscalar);
	rK2se= _mm_mul_pd(rK2e,rscalar);

	_mm_store_pd(K2d,rK2dr);
	_mm_store_pd(K2s,rK2sr);
	_mm_store_pd(K2d+2,rK2de);
	_mm_store_pd(K2s+2,rK2se);


	Z[indij+k  ] = exp(K2s[0])*K2d[0];
	Z[indij+k+1] = exp(K2s[1])*K2d[1];
	Z[indij+k+2] = exp(K2s[2])*K2d[2];
	Z[indij+k+3] = exp(K2s[3])*K2d[3];
      }
    }
  }
  SE_FGG_FREE(k1);SE_FGG_FREE(k2);SE_FGG_FREE(k3);
}

#else
// -----------------------------------------------------------------------------
// packing SE parameters

void scaling(double scalar, double *Z,
	     int n1, int n2, int n3, double* box)
{
  double *k1 = SE_FGG_MALLOC(sizeof(double)*n1);
  double *k2 = SE_FGG_MALLOC(sizeof(double)*n2);
  double *k3 = SE_FGG_MALLOC(sizeof(double)*n3);

   k_vec(n1,box,k1,k2,k3);

   int i,j,k,indij;
   double K2;	
   double Ki,Kij;
   double c = 4.*PI,d;
#ifdef _OPENMP
#pragma omp parallel for private(i,j,k,K2,Ki,Kij,indij)
#endif	
    for(i=0;i<n1;i++){
	  Ki = k1[i]*k1[i];
	  d = c/(double) (n1*n1*n1);
  	  for(j=0;j<n2;j++){
		Kij = Ki + k2[j]*k2[j];
		indij = i*n3*n2+j*n2;
		for(k=0;k<n3;k++){
			K2 = Kij + k3[k]*k3[k];
//			Z[indij+k] = exp(scalar*K2)/K2*4.*PI/(double) (n1*n1*n1);
			Z[indij+k] = exp(scalar*K2)/K2*d;
		}
	  }
     }
  SE_FGG_FREE(k1);SE_FGG_FREE(k2);SE_FGG_FREE(k3);
}
#endif

// -----------------------------------------------------------------------------
// packing SE parameters
void
SE_FGG_FCN_params(SE_FGG_params* params, const SE_opt* opt, int N)
{

    params->N = N;
    params->P = (int) opt->P;
    params->P_half=half( opt->P );
    params->c = opt->c;
    params->d = pow(params->c/PI,1.5);
    params->h = opt->box[0]/opt->M;
    params->a = -FGG_INF;

    params->dims[0] = opt->M;
    params->dims[1] = opt->M;
    params->dims[2] = opt->M;

    params->npdims[0] = params->dims[0]+params->P;
    params->npdims[1] = params->dims[1]+params->P;
    params->npdims[2] = params->dims[2]+params->P;

}

// -----------------------------------------------------------------------------
// interopolation and integrationto calculate forces
void
SE_fgg_int_force(double* x, double *q,double* H, int N, SE_opt opt, double* force_out)
{
  // pack parameters
  SE_FGG_params params;
  SE_FGG_FCN_params(&params, &opt, N);

  // scratch arrays
  SE_FGG_work work;
  SE_FGG_allocate_workspace(&work, &params,true,false);

  // coordinates and charges
  SE_state st = {.x = x, .q = q};
 
  if(VERBOSE)
	printf("[SE FG(G)] N=%d, P=%d\n",N,params.P);

   // now do the work
   SE_FGG_base_gaussian(&work, &params);
   SE_FGG_extend_fcn(&work, H, &params);
   
   // Integration
   SE_FGG_int_force(force_out, &work, &st, &params);
   
   // done  
   SE_FGG_free_workspace(&work);
}

// -----------------------------------------------------------------------------
// 3Dfft using fftw3 real to complex 
void do_fft_r2c_3d(double* in, t_complex* out, int n1, int n2, int n3)
{
  fftw_plan p;
  p = fftw_plan_dft_r2c_3d(n1, n2, n3, in, out, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}

// -----------------------------------------------------------------------------
// 3Dfft using fftw3 complex to real
void do_fft_c2r_3d(t_complex* in, double* out ,int n1, int n2, int n3)
{
  fftw_plan p;
  p = fftw_plan_dft_c2r_3d(n1, n2, n3, in, out, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}

// -----------------------------------------------------------------------------
// 3Dfft using fftw3 complex to complex
void do_fft_c2c_forward_3d(t_complex* in, t_complex* out ,int n1, int n2, int n3)
{
  fftw_plan p;
  p = fftw_plan_dft_3d(n1, n2, n3, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}

// -----------------------------------------------------------------------------
// 3Dfft using fftw3 complex to complex
void do_fft_c2c_backward_3d(t_complex* in, t_complex* out ,int n1, int n2, int n3)
{
  fftw_plan p;
  p = fftw_plan_dft_3d(n1, n2, n3, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}

// -----------------------------------------------------------------------------
// products sr(scalar to real). flag = 1 gives a.*b and flag = -1 gives a./b
void
product_sr(double* a, double scalar, double *c, int n1, int n2, int n3,int flag)
{
 int i;

switch (flag)
  {
   case 1:

#ifdef _OPENMP
#pragma omp parallel for private(i) shared(c)
#endif
	 for(i=0;i<n1*n2*n3;i++){
		c[i] = a[i]*scalar;
	 }
	 break;
   case -1:
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(c)
#endif
	 for(i=0;i<n1*n2*n3;i++){
		c[i] = a[i]/scalar;
	 }
   }

}

// -----------------------------------------------------------------------------
// product rr (real to real) equivalent to .* in MATLAB. 
// flag = 1 gives a.*b and flag = -1 gives a./b
void
product_rr(double* a, double* b, double *c, int n1, int n2, int n3,int flag)
{
 int i;

switch (flag)
  {
   case 1:

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	 for(i=0;i<n1*n2*n3;i++){
		c[i] = a[i]*b[i];
		c[i] = a[i]*b[i];
	 }
         break;
   case -1:
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	 for(i=0;i<n1*n2*n3;i++){
		c[i] = a[i]/b[i];
		c[i] = a[i]/b[i];
	 }
   }

}

// -----------------------------------------------------------------------------
// products rc (real to complex) equivalent to .* in MATLAB. 
// flag = 1 gives a.*b and flag = -1 gives a./b
void
product_rc(t_complex* a, double* b, t_complex *c, int n1,int n2, int n3,int flag)
{
 int i;

switch (flag)
  {
   case 1: 
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(c)
#endif
	 for(i=0;i<n1*n2*n3;i++){
		c[i][0] = a[i][0]*b[i];
		c[i][1] = a[i][1]*b[i];
	 }
         break;
   case -1:
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(c)
#endif
	 for(i=0;i<n1*n2*n3;i++){
		c[i][0] = a[i][0]/b[i];
		c[i][1] = a[i][1]/b[i];
	 }
   }

}


// -----------------------------------------------------------------------------
// printing results real 1d array while multiplying by c
void print_r1d(char* str, double *a, int n, double c,int verb)
{
  if(verb == 0)
	return;
  printf("%s:\n",str);
  int i;
  for (i=0;i<n;i++)
	printf("%.12f\n",a[i]*c);
  printf("\n");
}

// -----------------------------------------------------------------------------
// printing results complex 1d array while multiplying by c
void print_c1d(char* str, t_complex *a, int n, double c,int verb)
{
  if (verb == 0)
 	return;
  printf("%s:\n",str);
  int i;
  for (i=0;i<n;i++)
	printf("(%.4f,%.4f)\t",a[i][0]*c,a[i][1]*c);
  printf("\n");
}

// -----------------------------------------------------------------------------
// printing results real 3d array while multiplying by c
void print_r3d(char* str, double *a, int n1, int n2, int n3, double c,int verb)
{
  if(verb == 0)
	return;
  printf("%s:\n",str);
  int i,j,k;
  for (i=0;i<n1;i++){
	for(j=0;j<n2;j++){
		for(k=0;k<n3;k++)
			printf("%3.4f\t",a[i*n3*n2+j*n2+k]*c);
		printf("\n");
	}
	printf("\n");
   }
   printf("\n");
}

// -----------------------------------------------------------------------------
// printing results complex 3d array while multiplying by c
void print_c3d(char* str, t_complex *a, int n1, int n2, int n3, double c,int verb)
{
  if (verb == 0)
 	return;
  printf("%s:\n",str);
  int i,j,k;
  for (i=0;i<n1;i++){
	for(j=0;j<n2;j++){
		for(k=0;k<n3;k++)
			printf("(%3.4f,%3.4f)\t",a[i*n3*n2+j*n2+k][0]*c,a[i*n3*n2+j*n2+k][1]*c);
		printf("\n");
	}
	printf("\n");
   }
   printf("\n");
}

// -----------------------------------------------------------------------------
// copy real vec into complex vec

void
copy_r2c(double* in, t_complex* out,int n)
{
  int i;
  for (i=0;i<n;i++){
    out[i][0] = in[i];
    out[i][1] = 0.;
  }
}


// -----------------------------------------------------------------------------
// copy complex vec into real vec

void
copy_c2r(t_complex* in, double* out,int n)
{
  int i;
  for (i=0;i<n;i++)
    out[i] = in[i][0];

}


// ----------------------------------------------------------------------------
// lambertw function similar to MATLAB
/* written K M Briggs Keith dot Briggs at bt dot com 97 May 21.  
   Revised KMB 97 Nov 20; 98 Feb 11, Nov 24, Dec 28; 99 Jan 13; 00 Feb 23; 01 Apr 09

   Computes Lambert W function, principal branch.
   See LambertW1.c for -1 branch.

   Returned value W(z) satisfies W(z)*exp(W(z))=z
   test data...
      W(1)= 0.5671432904097838730
      W(2)= 0.8526055020137254914
      W(20)=2.2050032780240599705
   To solve (a+b*R)*exp(-c*R)-d=0 for R, use
   R=-(b*W(-exp(-a*c/b)/b*d*c)+a*c)/b/c
*/

double 
LambertW(const double z) {
  int i; 
  const double eps=4.0e-16, em1=0.3678794411714423215955237701614608; 
  double p,e,t,w;
  if (z<-em1 || isinf(z) || isnan(z)) { 
    fprintf(stderr,"LambertW: bad argument %g, exiting.\n",z); exit(1); 
  }
  if (0.0==z) return 0.0;
  if (z<-em1+1e-4) { // series near -em1 in sqrt(q)
    double q=z+em1,r=sqrt(q),q2=q*q,q3=q2*q;
    return 
     -1.0
     +2.331643981597124203363536062168*r
     -1.812187885639363490240191647568*q
     +1.936631114492359755363277457668*r*q
     -2.353551201881614516821543561516*q2
     +3.066858901050631912893148922704*r*q2
     -4.175335600258177138854984177460*q3
     +5.858023729874774148815053846119*r*q3
     -8.401032217523977370984161688514*q3*q;  // error approx 1e-16
  }
  /* initial approx for iteration... */
  if (z<1.0) { /* series near 0 */
    p=sqrt(2.0*(2.7182818284590452353602874713526625*z+1.0));
    w=-1.0+p*(1.0+p*(-0.333333333333333333333+p*0.152777777777777777777777)); 
  } else 
    w=log(z); /* asymptotic */
  if (z>3.0) w-=log(w); /* useful? */
  for (i=0; i<10; i++) { /* Halley iteration */
    e=exp(w); 
    t=w*e-z;
    p=w+1.0;
    t/=e*p-0.5*(p+1.0)*t/p; 
    w-=t;
    if (fabs(t)<eps*(1.0+fabs(w))) return w; /* rel-abs error */
  }
  /* should never get here */
  fprintf(stderr,"LambertW: No convergence at z=%g, exiting.\n",z); 
  exit(1);
}



double
SE_init_params(int* M,int* P,double* m, double eps, double L,
               double xi, const double* q, int N)
{

  int i,iP,M_l;
  double m_l, App,Q=0.,c=.95, max_q=0;

  for (i=0;i<N;i++){
	Q += q[i]*q[i];
	max_q = (max_q<q[i]) ? q[i] : max_q;
  }

#ifdef POTENTIAL
  iP = *P;
  // based on Lindbo & Tornberg error estimate
  m_l = c*sqrt(PI*iP);
  App = sqrt(Q*xi*L)/L*exp(-PI*iP*c*c/2);
#endif

 // Based on Kolafa & Perram error estimate
  // to remove the truncation error
  double k_inf = 2.*xi*L;
 M_l = ceil(2.*k_inf+1)+2;

#ifdef FORCE
  iP = *P;
  // based on Lindbo & Tornberg error estimate
  m_l = c*sqrt(PI*iP);

  App = sqrt(Q*xi*L)/L*exp(-PI*iP*c*c/2);
  printf("bound for potential : %g\n",App);
  App = Q*sqrt(xi*L)/L/L*exp(-PI*iP*c*c/2.)*2.*PI*c*c*iP;
  printf("bound for force: %g\n",App);

#endif



 if(M_l<iP)
   M_l = iP;
 
 *m = m_l;
 *M = M_l;
 return Q;
}
