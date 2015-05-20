/* The code is written by Dag Lindbo. The fourier direct sum 
 * is edited by davoud Saffar to reduce to number of operations
 * by a factor of 8.
*/

#include "SE_direct.h"
#include "math.h"
#include "SE_fgg.h"
#include "SE_general.h"


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


/* This code is edite by Davoud Saffar
 * to reduce the number of loops by an order of 8
*/
void SE3P_direct(double* restrict phi, 
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


void SE3P_direct_force(double* restrict force,
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

//		    double h = hs(j0,j1,j2);
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


void SE3P_direct_real(double* restrict phi,
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

