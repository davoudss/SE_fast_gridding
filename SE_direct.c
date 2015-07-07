/* The SE3P and SE2P routines are written by Dag Lindbo, dag@kth.se. 
 * The fourier direct sum is edited to reduce 
 * the number of operations by a factor of 8.
 * SE1P routines are added.
 * Davoud Saffar Shamshirgar davoudss@kth.se
 */

#include "SE_direct.h"
#include <math.h>
#include "SE_fgg.h"
#include "SE_general.h"
#include "mathint.h"
#define __EXP_ARG_MAX 600

#ifdef ONE_PERIODIC
void SE1P_direct_real(double* restrict phi, 
		      const double* restrict x, 
		      const double* restrict q, int N,
		      const ewald_opts opt)
{
  double rvec[3];
  double qn;
  double p, r;
  for(int m=0; m<N; m++)
    {
      p = 0;
      for(int n=0; n<N; n++)
	{
	  rvec[0] = x[m    ]-x[n    ];
	  rvec[1] = x[m+N  ]-x[n+N  ];
	  rvec[2] = x[m+2*N]-x[n+2*N];
	  qn = q[n];

	  for(int p2 = -opt.layers; p2<=opt.layers; p2++)
	    {
	      if(m == n && p2 == 0)
		continue;
			
	      r = sqrt((rvec[2]+p2*opt.box[2])*
		       (rvec[2]+p2*opt.box[2])+
		       rvec[0]*rvec[0]+
		       rvec[1]*rvec[1]);
			
	      p += qn*erfc(opt.xi*r)/r;
	    }
	}
      phi[m] = p;
    }
}



void SE1P_direct_real_rc(double* restrict phi, 
			 const double* restrict x, 
			 const double* restrict q, int N,
			 const ewald_opts opt)
{
  double rvec[3];
  double qn;
  double p, r;
  for(int m=0; m<N; m++)
    {
      p = 0;
      for(int n=0; n<N; n++)
	{
	  rvec[0] = x[m    ]-x[n    ];
	  rvec[1] = x[m+N  ]-x[n+N  ];
	  rvec[2] = x[m+2*N]-x[n+2*N];
	  qn = q[n];

	  for(int p2 = -opt.layers; p2<=opt.layers; p2++)
	    {
	      if(m == n && p2 == 0)
		continue;
			
	      r = sqrt((rvec[2]+p2*opt.box[2])*
		       (rvec[2]+p2*opt.box[2])+
		       rvec[0]*rvec[0]+
		       rvec[1]*rvec[1]);
			
	      if(r > opt.rc) continue;

	      p += qn*erfc(opt.xi*r)/r;
	    }
	}
      phi[m] = p;
    }
}

void SE1P_direct_fd(double* restrict phi, 
		    const double* restrict x, 
		    const double* restrict q, int N,
		    const ewald_opts opt)
{
  double xm[3],M; 
  double k3, k3z, z, phi_m, rho2, a, b, K0,qn;

  const double xi   = opt.xi;
  double xi2        = xi*xi;
  double TwoPiOverL = 2.*PI/opt.box[2];
  int xiL = xi*opt.box[2];
  
  // prepare for integration

  M         = 1000;                                  // Initial guess
  int M1    = MIN(700,MAX(xiL,200));                 
  int M2    = MIN(M-M1,300);
  // if M1+M2 is not perfect divisior of 4 then make it
  int di    = (M1+M2)%4; di = di%4;
  M2        = M2 + di;                               // remainder added to M2
  double v  = 1./(xi*opt.box[2]);	             // splitting point
  M         = M1+M2; // update the guess
  double *X = (double*) _mm_malloc(M*sizeof(double),32); // quadrature points
  double *W = (double*) _mm_malloc(M*sizeof(double),32); // quadrature weights

  glwt_fast(M1,0.,v,X,W);
  glwt_fast(M2,v,1.,&X[M1],&W[M1]);

  for(int m=0; m<N; m++)
    {
      xm[0] = x[m    ];
      xm[1] = x[m+N  ];
      xm[2] = x[m+2*N];
      phi_m = 0;
      for(int n = 0; n<N; n++)
	{
	  z   = xm[2]-x[n+2*N];
	  rho2= ( (xm[0]-x[n  ])*(xm[0]-x[n  ])+
	          (xm[1]-x[n+N])*(xm[1]-x[n+N]) );
	  b   = rho2*xi2;
	  qn  = q[n];
	  for(int j2 = -opt.layers; j2<=opt.layers; j2++)
	    {
	      if(j2 == 0)
		continue;
	      
	      k3  = TwoPiOverL*j2;
	      k3z = -k3*z;
	      
	      a   = k3*k3/(4.*xi2);
	      K0  = IncompBesselK0_int(a,b,M,X,W);
	      phi_m += qn*cos(k3z)*K0;
	    }
	}
      phi[m] = phi_m/(opt.box[2]);
    }
}


void SE1P_direct_k0(double* restrict phi, 
		    const double* restrict x, 
		    const double* restrict q, int N,
		    const ewald_opts opt)
{
  double phi_m, rho2,xm[2];
  const double xi = opt.xi;

  for(int m=0; m<N; m++)
    {
      phi_m=0;
      xm[0] = x[m    ];
      xm[1] = x[m+N  ];
      for(int n=0; n<N; n++)
	{
	  if(m==n)
	    continue;
	  rho2 = ( (xm[0]-x[n  ])*(xm[0]-x[n  ]) + 
		   (xm[1]-x[n+N])*(xm[1]-x[n+N]) );
	  phi_m += -q[n]*(EULER + log(rho2*xi*xi) + expint(1,rho2*xi*xi));
	}
      phi[m] = phi_m/opt.box[2];
    }
}

void SE1P_direct_self(double* restrict phi, 
		      const double* restrict q, int N, 
		      const ewald_opts opt)
{
  double c = 2*opt.xi/sqrt(PI);
  for(int m=0; m<N; m++)
    phi[m] = -c*q[m];
} 

#elif defined ( TWO_PERIODIC )
void SE2P_direct_real(double* restrict phi, 
		      const double* restrict x, 
		      const double* restrict q, int N,
		      const ewald_opts opt)
{
  double rvec[3];
  double qn;
  double p, r;
  for(int m=0; m<N; m++)
    {
      p = 0;
      for(int n=0; n<N; n++)
	{
	  rvec[0] = x[m    ]-x[n    ];
	  rvec[1] = x[m+N  ]-x[n+N  ];
	  rvec[2] = x[m+2*N]-x[n+2*N];
	  qn = q[n];

	  for(int p0 = -opt.layers; p0<=opt.layers; p0++)
	    for(int p1 = -opt.layers; p1<=opt.layers; p1++)
	      {
		if(m == n && p1 == 0 && p0 == 0)
		  continue;
			
		r = sqrt((rvec[0]+p0*opt.box[0])*
			 (rvec[0]+p0*opt.box[0])+
			 (rvec[1]+p1*opt.box[1])*
			 (rvec[1]+p1*opt.box[1])+
			 rvec[2]*rvec[2]);
			
		p += qn*erfc(opt.xi*r)/r;
	      }
	}
      phi[m] = p;
    }
}

void SE2P_direct_real_rc(double* restrict phi, 
			 const double* restrict x, 
			 const double* restrict q, int N,
			 const ewald_opts opt)
{
  double rvec[3];
  double qn;
  double p, r;
  for(int m=0; m<N; m++)
    {
      p = 0;
      for(int n=0; n<N; n++)
	{
	  rvec[0] = x[m    ]-x[n    ];
	  rvec[1] = x[m+N  ]-x[n+N  ];
	  rvec[2] = x[m+2*N]-x[n+2*N];
	  qn = q[n];

	  for(int p0 = -opt.layers; p0<=opt.layers; p0++)
	    for(int p1 = -opt.layers; p1<=opt.layers; p1++)
	      {
		if(m == n && p1 == 0 && p0 == 0)
		  continue;
			
		r = sqrt((rvec[0]+p0*opt.box[0])*
			 (rvec[0]+p0*opt.box[0])+
			 (rvec[1]+p1*opt.box[1])*
			 (rvec[1]+p1*opt.box[1])+
			 rvec[2]*rvec[2]);
			
		if(r > opt.rc) continue;

		p += qn*erfc(opt.xi*r)/r;
	      }
	}
      phi[m] = p;
    }
}

static inline double theta_plus(double z, double k, double xi)
{
  /* idea for a more stable form [LK] */
  /* exp( k*z + log( erfc(k/(2.0*xi) + xi*z) ) ); */

  if(k*z <  __EXP_ARG_MAX)
    return exp( k*z)*erfc(k/(2.0*xi) + xi*z);
  else 
    return 0.0;
}

static inline double theta_minus(double z, double k, double xi)
{
  /* exp(-k*z + log( erfc(k/(2.0*xi) - xi*z) ) ); */

  if(-k*z <  __EXP_ARG_MAX)
    return exp(-k*z)*erfc(k/(2.0*xi) - xi*z);
  else 
    return 0.0;
}

void SE2P_direct_fd(double* restrict phi, 
		    const double* restrict x, 
		    const double* restrict q, int N,
		    const ewald_opts opt)
{
  double k[2], xm[3]; 
  double kn, k_dot_r, z, phi_m;
  double cm, cp;
  const double xi = opt.xi;

  for(int m=0; m<N; m++)
    {
      xm[0] = x[m    ];
      xm[1] = x[m+N  ];
      xm[2] = x[m+2*N];
      phi_m = 0;
      for(int n = 0; n<N; n++){
	for(int j0 = -opt.layers; j0<=opt.layers; j0++)
	  for(int j1 = -opt.layers; j1<=opt.layers; j1++)
	    {
	      if(j0 == 0 && j1 == 0)
		continue;

	      k[0] = 2*PI*j0/opt.box[0];
	      k[1] = 2*PI*j1/opt.box[1];
	      kn = sqrt(k[0]*k[0] + k[1]*k[1]);
	      k_dot_r = k[0]*(xm[0]-x[n]) + k[1]*(xm[1]-x[n+N]);
	      z = xm[2]-x[n+2*N];
	      cp = theta_plus(z,kn,xi);
	      cm = theta_minus(z,kn,xi);

	      phi_m += q[n]*cos(k_dot_r)*(cm+cp)/kn;
	    }
      }
      phi[m] = PI*phi_m/(opt.box[0]*opt.box[1]);
    }
}

void SE2P_direct_k0(double* restrict phi, 
		    const double* restrict x, 
		    const double* restrict q, int N,
		    const ewald_opts opt)
{
  double z, zm, phi_m;
  for(int m=0; m<N; m++)
    {
      phi_m=0;
      zm = x[m+2*N];
      for(int n=0; n<N; n++)
	{
	  z = zm - x[n+2*N];
	  phi_m += q[n]*(exp(-opt.xi*opt.xi*z*z)/opt.xi + 
			 sqrt(PI)*z*erf(opt.xi*z));
	}
      phi[m] = -2*phi_m*sqrt(PI)/(opt.box[0]*opt.box[1]);
    }
}

void SE2P_direct_self(double* restrict phi, 
		      const double* restrict q, int N, 
		      const ewald_opts opt)
{
  double c = 2*opt.xi/sqrt(PI);
  for(int m=0; m<N; m++)
    phi[m] = -c*q[m];
}


#else

double
heaviside(int x)
{
  if(x==0)
    return .5;
  else
    return 1.;
}

double 
hs(int x, int y, int z)
{
  double h=heaviside(x)*heaviside(y)*heaviside(z);
  if(h==1)
    return 8.;
  else if(h==.5)
    return 4.;
  else
    return 2;
}


/* This code is edited by Davoud Saffar
 * to reduce the number of loops by an order of 8
 */
void SE3P_direct_fd(double* restrict phi, 
		    const double* restrict x, 
		    const double* restrict q, int N,
		    const ewald_opts opt)
{
  double k[3];
  double k2, z, p;
  double c = 4.*PI/(opt.box[0]*opt.box[1]*opt.box[2]);
   
#ifdef _OPENMP
#pragma omp parallel for private(k,k2,z,p)
#endif
  for(int m=0; m<N; m++)
    {
      p = 0;
      for(int j0 = 0; j0<=opt.layers; j0++)
	for(int j1 = 0; j1<=opt.layers; j1++)
	  for(int j2 = 0; j2<=opt.layers; j2++)
	    {
	      if(j0 == 0 && j1 == 0 && j2==0)
		continue;
	      k[0] = 2*PI*j0/opt.box[0];
	      k[1] = 2*PI*j1/opt.box[1];
	      k[2] = 2*PI*j2/opt.box[2];
	      k2 = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
	
	      z=0;
	      for(int n = 0; n<N; n++)
		z += q[n]*cos(k[0]*(x[m]-x[n]))*
		  cos(k[1]*(x[m+N]-x[n+N]))*
		  cos(k[2]*(x[m+2*N]-x[n+2*N]));

	      double h = hs(j0,j1,j2);
	      p += h*z*exp(-k2/(4.*opt.xi*opt.xi))/k2;
	    }
      phi[m] = c*p;
    }
}


void SE3P_direct_force_fd(double* restrict force,
			  const double* restrict x, 
			  const double* restrict q, int N,
			  const ewald_opts opt)
{
  double k[3];
  double k2, z, p[3];
  double c = 4.*PI/(opt.box[0]*opt.box[1]*opt.box[2]);
   
#ifdef _OPENMP
#pragma omp parallel for private(k,k2,z,p)
#endif
  for(int m=0; m<N; m++)
    {
      p[0] = 0.; p[1]=0.; p[2]=0.;
      for(int j0 = -opt.layers; j0<=opt.layers; j0++)
	for(int j1 = -opt.layers; j1<=opt.layers; j1++)
	  for(int j2 = -opt.layers; j2<=opt.layers; j2++)
	    {
	      if(j0 == 0 && j1 == 0 && j2==0)
		continue;
	      k[0] = 2*PI*j0/opt.box[0];
	      k[1] = 2*PI*j1/opt.box[1];
	      k[2] = 2*PI*j2/opt.box[2];
	      k2 = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
	
	      z=0;
	      for(int n = 0; n<N; n++){
		z += q[n]*sin(k[0]*(x[n    ]-x[m    ])+
			      k[1]*(x[n+  N]-x[m+  N])+
			      k[2]*(x[n+2*N]-x[m+2*N]));
	      }

	      double h = 1.;

	      p[0] += h*z*exp(-k2/(4.*opt.xi*opt.xi))/k2*k[0];
	      p[1] += h*z*exp(-k2/(4.*opt.xi*opt.xi))/k2*k[1];
	      p[2] += h*z*exp(-k2/(4.*opt.xi*opt.xi))/k2*k[2];
	    }

      force[m    ] = .5*c*p[0]*q[m];
      force[m+  N] = .5*c*p[1]*q[m];
      force[m+2*N] = .5*c*p[2]*q[m];
    }
}


void SE3P_direct_real_rc(double* restrict phi,
                         const double* restrict x,
                         const double* restrict q, int N,
                         const ewald_opts opt)
{
  double rvec[3];
  double qn;
  double p, r;
  int m,n;
#ifdef _OPENMP
#pragma omp parallel for private(rvec, qn, p, r,m,n)
#endif
  for(m=0; m<N; m++)
    {
      p = 0;
      for(n=0; n<N; n++)
	{
	  rvec[0] = x[m    ]-x[n    ];
	  rvec[1] = x[m+N  ]-x[n+N  ];
	  rvec[2] = x[m+2*N]-x[n+2*N];
	  qn = q[n];

	  for(int p0 = -opt.layers; p0<=opt.layers; p0++)
	    for(int p1 = -opt.layers; p1<=opt.layers; p1++)
	      for(int p2 = -opt.layers; p2<=opt.layers; p2++)
		{
		  if(m == n && p2 == 0 && p1 == 0 && p0 == 0)
		    continue;

		  r = sqrt((rvec[0]+p0*opt.box[0])*
			   (rvec[0]+p0*opt.box[0])+
			   (rvec[1]+p1*opt.box[1])*
			   (rvec[1]+p1*opt.box[1])+
			   (rvec[2]+p2*opt.box[2])*
			   (rvec[2]+p2*opt.box[2]));
		  if(r > opt.rc) continue;

		  p += qn*erfc(opt.xi*r)/r;
		}
	}
      phi[m] += p;
    }
}

void SE3P_direct_real(double* restrict phi, 
		      const double* restrict x, 
		      const double* restrict q, int N,
		      const ewald_opts opt)
{
  double rvec[3];
  double qn;
  double p, r;
#ifdef _OPENMP
#pragma omp parallel for private(rvec, qn, p, r)
#endif
  for(int m=0; m<N; m++)
    {
      p = 0;
      for(int n=0; n<N; n++)
	{
	  rvec[0] = x[m    ]-x[n    ];
	  rvec[1] = x[m+N  ]-x[n+N  ];
	  rvec[2] = x[m+2*N]-x[n+2*N];
	  qn = q[n];

	  for(int p0 = -opt.layers; p0<=opt.layers; p0++)
	    for(int p1 = -opt.layers; p1<=opt.layers; p1++)
	      for(int p2 = -opt.layers; p2<=opt.layers; p2++)
		{
		  if(m == n && p2 == 0 && p1 == 0 && p0 == 0)
		    continue;
			
		  r = sqrt((rvec[0]+p0*opt.box[0])*
			   (rvec[0]+p0*opt.box[0])+
			   (rvec[1]+p1*opt.box[1])*
			   (rvec[1]+p1*opt.box[1])+
			   (rvec[2]+p2*opt.box[2])*
			   (rvec[2]+p2*opt.box[2]));
			
		  p += qn*erfc(opt.xi*r)/r;
		}
	}
      phi[m] += p;
    }
}


void SE3P_direct_self(double* restrict phi,
		      const double* restrict q, int N,
		      const ewald_opts opt)
{
  double c = 2*opt.xi/sqrt(PI);
  int m;
#ifdef _OPENMP
#pragma omp parallel for private(m)
#endif
  for(m=0; m<N; m++)
    phi[m] -= c*q[m];
}

#endif

