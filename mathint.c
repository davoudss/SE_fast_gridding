#include "mathint.h"

#if (defined __AVX__ || defined __SSE4_2__)
#include "math_x86.h"
#endif


double rk2_leg ( double t1, double t2, double x, int n )
/*
  RK2_LEG advances the value of X(T) using a Runge-Kutta method.
*/
{
  double f,h,k1,k2,snn1,t;
  int j,m=10;

  h = ( t2 - t1 ) / ( double ) m;
  snn1 = sqrt ( ( double ) ( n * ( n + 1 ) ) );

  t = t1;

  for ( j = 0; j < m; j++ )
    {
      f = ( 1.0 - x ) * ( 1.0 + x );
      k1 = - h * f / ( snn1 * sqrt ( f ) - 0.5 * x * sin ( 2.0 * t ) );
      x = x + k1;

      t = t + h;

      f = ( 1.0 - x ) * ( 1.0 + x );
      k2 = - h * f / ( snn1 * sqrt ( f ) - 0.5 * x * sin ( 2.0 * t ) );   
      x = x + 0.5 * ( k2 - k1 );
    }
  return x;
}

double ts_mult ( double *u, double h, int n )
/*
  TS_MULT evaluates a polynomials
*/
{
  double hk,ts;
  int k;
  
  ts = 0.0;
  hk = 1.0;
  for ( k = 1; k<= n; k++ )
    {
      ts = ts + u[k] * hk;
      hk = hk * h;
    }
  return ts;
}


void legendre_compute_glr0 ( int n, double *p, double *pp )

/*
  LEGENDRE_COMPUTE_GLR0 gets a starting value for the fast algorithm.
*/
{
  double dk;
  int k;
  double pm1,pm2,ppm1,ppm2;

  pm2 = 0.0;
  pm1 = 1.0;
  ppm2 = 0.0;
  ppm1 = 0.0;

  for ( k = 0; k < n; k++ )
    {
      dk = ( double ) k;
      *p = - dk * pm2 / ( dk + 1.0 );
      *pp = ( ( 2.0 * dk + 1.0 ) * pm1 - dk * ppm2 ) / ( dk + 1.0 );
      pm2 = pm1;
      pm1 = *p;
      ppm2 = ppm1;
      ppm1 = *pp;
    }
  return;
}

void legendre_compute_glr1 ( int n, double *x, double *ders )
/*
  LEGENDRE_COMPUTE_GLR1 gets the complete set of Legendre points and weights.
*/
{
  double dk,dn,h;
  int j,k,l,m=30,n2,s;

  double *u;
  double *up;
  double xp;

  if ( n % 2 == 1 )
    {
      n2 = ( n - 1 ) / 2;
      s = 1;
    }
  else
    {
      n2 = n / 2;
      s = 0;
    }

  u  = ( double * ) _mm_malloc ( ( m + 2 ) * sizeof ( double ) ,32);
  up = ( double * ) _mm_malloc ( ( m + 1 ) * sizeof ( double ) ,32);

  dn = ( double ) n;

  for ( j = n2; j < n - 1; j++ )
    {
      xp = x[j];

      h = rk2_leg ( pi/2.0, -pi/2.0, xp, n ) - xp;

      u[0] = 0.0;
      u[1] = 0.0;
      u[2] = ders[j];

      up[0] = 0.0;
      up[1] = u[2];

      for ( k = 0; k <= m - 2; k++ )
	{
	  dk = ( double ) k;

	  u[k+3] = 
	    ( 
	     2.0 * xp * ( dk + 1.0 ) * u[k+2]
	     + ( dk * ( dk + 1.0 ) - dn * ( dn + 1.0 ) ) * u[k+1] / ( dk + 1.0 )
	      ) / ( 1.0 - xp ) / ( 1.0 + xp ) / ( dk + 2.0 );

	  up[k+2] = ( dk + 2.0 ) * u[k+3];
	}

      for ( l = 0; l < 5; l++ )
	{ 
	  h = h - ts_mult ( u, h, m ) / ts_mult ( up, h, m-1 );
	}

      x[j+1] = xp + h;
      ders[j+1] = ts_mult ( up, h, m-1 );
    }

  free ( u );
  free ( up );

  for ( k = 0; k < n2 + s; k++ )
    {
      x[k] = - x[n-k-1];
      ders[k] = ders[n-k-1];
    }
  return;
}

void legendre_compute_glr2 ( double pn0, int n, double *x1,  double *d1 )
/*
  LEGENDRE_COMPUTE_GLR2 finds the first real root.
*/
{
  double dk,dn,t;
  int k,l,m=30;

  double *u;
  double *up;

  t = 0.0;
  *x1 = rk2_leg ( t, -pi/2.0, 0.0, n );

  u  = ( double * ) _mm_malloc ( ( m + 2 ) * sizeof ( double ) ,32);
  up = ( double * ) _mm_malloc ( ( m + 1 ) * sizeof ( double ) ,32);

  dn = ( double ) n;
  /*
    U[0] and UP[0] are never used.
    U[M+1] is set, but not used, and UP[M] is set and not used.
    What gives?
  */
  u[0] = 0.0;
  u[1] = pn0;

  up[0] = 0.0;
 
  for ( k = 0; k <= m - 2; k = k + 2 )
    {
      dk = ( double ) k;

      u[k+2] = 0.0;
      u[k+3] = ( dk * ( dk + 1.0 ) - dn * ( dn + 1.0 ) ) * u[k+1]
	/ ( dk + 1.0 ) / ( dk + 2.0 );
 
      up[k+1] = 0.0;
      up[k+2] = ( dk + 2.0 ) * u[k+3];
    }
  
  for ( l = 0; l < 5; l++ )
    {
      *x1 = *x1 - ts_mult ( u, *x1, m ) / ts_mult ( up, *x1, m-1 );
    }
  *d1 = ts_mult ( up, *x1, m-1 );

  free ( u );
  free ( up) ;

  return;
}


void legendre_compute_glr ( int n, double x[], double w[] )
/*
  LEGENDRE_COMPUTE_GLR: Legendre quadrature by the Glaser-Liu-Rokhlin method.
*/
{
  int i;
  double p=0;
  double pp=0;
  double w_sum;
  /*
    Get the value and derivative of the N-th Legendre polynomial at 0.
  */
  legendre_compute_glr0 ( n, &p, &pp );
  /*
    Either zero is a root, or we have to call a function to find the first root.
  */  
  if ( n % 2 == 1 )
    {
      x[(n-1)/2] = p;
      w[(n-1)/2] = pp;
    }
  else
    {
      legendre_compute_glr2 ( p, n, &x[n/2], &w[n/2] );
    }
  /*
    Get the complete set of roots and derivatives.
  */
  legendre_compute_glr1 ( n, x, w );
  /*
    Compute the weights.
  */
  for ( i = 0; i < n; i++ )
    {
      w[i] = 2.0 / ( 1.0 - x[i] ) / ( 1.0 + x[i] ) / w[i] / w[i];
    }
  w_sum = 0.0;
  for ( i = 0; i < n; i++ )
    {
      w_sum = w_sum + w[i];
    }
  for ( i = 0; i < n; i++ )
    {
      w[i] = 2.0 * w[i] / w_sum;
    }
  return;
}

void rescale ( double a, double b, int n, double x[], double w[] )
{
  int i;

  for ( i = 0; i < n; i++ )
    {
      x[i] = ( ( a + b ) + ( b - a ) * x[i] ) / 2.0;
    }
  for ( i = 0; i < n; i++ )
    {
      w[i] = ( b - a ) * w[i] / 2.0;
    }
  return;
}

void glwt_fast( int n, double a, double b , double *x, double* w)
/*
  LEGENDRE_HANDLE computes the requested Gauss-Legendre rule and outputs it.
*/ 
{
  /*
    Compute the rule.
  */
  legendre_compute_glr ( n, x, w );
  /*
    Rescale the rule to [A,B].
  */
  rescale ( a, b, n, x, w );
}

void glwt(int n, double a, double b, double *x, double *w)
{
  int i,k,N,N1,N2;
  double h,d,H;
  N  = n - 1;
  N1 = N + 1; 
  N2 = N + 2;
  
  double *xu = (double*) _mm_malloc(N1*sizeof(double),32);
  double *y  = (double*) _mm_malloc(N1*sizeof(double),32);
  double *L  = (double*) _mm_malloc(N1*sizeof(double)*3,32);
  double *Lp = (double*) _mm_malloc(N1*sizeof(double),32);
  // Initial guess
  h = 2./(double) (N1-1.);
  for (i=0;i<N1;i++)
    {
      xu[i]   = -1. + i*h;
      y[i] = cos( (2.*i+1.)*pi/(2.*N+2.) )+ 
	(0.27)/(double) N1*sin(pi*xu[i]*N/(double) N2);
    }
  d = 2;
  while (d>EPS)
    {
      for(i=0;i<N1;i++)
	{
	  L[i] = 1.;
	  L[N1+i] = y[i];
	}
      for (k=1;k<N1-1;k++)
	for(i=0;i<N1;i++)
	  {
	    L[2*N1+i] = ( (2.*k+1.)*y[i]*L[N1+i]-k*L[i] )/(double) (k+1.);
	    L[i]    = L[N1+i];
	    L[N1+i] = L[2*N1+i];
	  }
      // last iteration remained
      k = N1-1;
      for(i=0;i<N1;i++)
	{
	  L[2*N1+i] = ( (2.*k+1.)*y[i]*L[N1+i]-k*L[i] )/(double) (k+1.);
	}
      d = 0;
      for(i=0;i<N1;i++)
	{
	  Lp[i] = N2*( L[N1+i]-y[i]*L[2*N1+i]  )/( 1-y[i]*y[i]  );
	  H = L[2*N1+i]/Lp[i];
	  d = MAX(d,fabs(H));
	  y[i] -= H;
	}
    }
 
  // Linear map from [-1,1] to [a,b]
  double c = N2*N2/(double) (N1*N1);
  for (i=0;i<N1;i++)
    {
      x[N1-i-1] = ( a*(1.-y[i])+b*(1.+y[i]) )/2.;
      w[N1-i-1] = (b-a)/( (1.-y[i]*y[i])*Lp[i]*Lp[i] )*c;
    }
}

void trapz_wt(int n, double *x, double *w)
{
  int i;
  double h = 1./(double) (n-1);
 
  for (i=0;i<n;i++)
    {
      x[i] = i*h;
      w[i] = h;
    }
  w[0  ] /=2.;
  w[n-1] /=2.;
}

static inline double K0(double a, double b, double x)
{
  if(x==0)
    return 0.;
  else
    return 1./x*exp(-b*x-a/x);
}

double IncompBesselK0_int(double a, double b, int n,double *x, double*w)
{
#ifdef __AVX__
  int i;
  double s[4] __attribute__((aligned(32)));
  
  __m256d S,X,C,A,B,AX,BX,ONE,InvX,ExpC,W;
  S = _mm256_setzero_pd();

  for (i=0;i<n;i+=4)
    {
      X    = _mm256_load_pd( x + i );
      B    = _mm256_set1_pd( -b );
      A    = _mm256_set1_pd( -a );
      AX   = _mm256_div_pd( A, X  );
      BX   = _mm256_mul_pd( B, X  );
      C    = _mm256_add_pd( AX, BX  );
      ONE  = _mm256_set1_pd( 1.0 );
      InvX = _mm256_div_pd(ONE , X);
      ExpC = se_mm256_exp_pd( C );
      W    = _mm256_load_pd( w + i );
      S    = _mm256_add_pd(S, _mm256_mul_pd( W, _mm256_mul_pd(InvX, ExpC) ));
    }
  _mm256_store_pd(s, S);
  return s[0]+s[1]+s[2]+s[3];
#elif defined __SSE4_2__
  int i;
  double s[2] __attribute__((aligned(16)));
  
  __m128d S,X,C,A,B,AX,BX,ONE,InvX,ExpC,W;
  S = _mm_setzero_pd();

  for (i=0;i<n;i+=2)
    {
      X    = _mm_load_pd( x + i );
      B    = _mm_set1_pd( -b );
      A    = _mm_set1_pd( -a );
      AX   = _mm_div_pd( A, X  );
      BX   = _mm_mul_pd( B, X  );
      C    = _mm_add_pd( AX, BX  );
      ONE  = _mm_set1_pd( 1.0 );
      InvX = _mm_div_pd(ONE , X);
      ExpC = se_mm_exp_pd( C );
      W    = _mm_load_pd( w + i );
      S    = _mm_add_pd(S, _mm_mul_pd( W, _mm_mul_pd(InvX, ExpC  ) ));
    }
  _mm_store_pd(s, S);
  return s[0]+s[1];
#else
  int i;
  double s=0.,f;
  for (i=0;i<n;i++)
    {
      f = K0(a,b,x[i]);
      s+= w[i]*f;
    }
  return s;
#endif
}


double IncompBesselK0_int_inf(double a, double b, int n)
{
  int i;
  double eeps = 1.e-15;
  double aa = log(1./eeps)/a;
  double bb = b/a;
  
  unsigned long int tmax = aa/2+sqrt(aa*aa/4-bb);
  double h = (double) (tmax-1.)/n;
  
  double *tvec = (double*) _mm_malloc(sizeof(double)*(n+1),32);
  double *wv   = (double*) _mm_malloc(sizeof(double)*(n+1),32);
  for (i=0; i<=n;i++)
    {
      tvec[i] = 1.+i*h;
      wv[i]   = 1.;
    }
  wv[0] = .5; wv[n] = .5;
  
  double  corrv[5] ={-245./1440., 
		     462./1440.,
		     -336./1440.,
		     146./1440.,
		     -27./1440.};
  
  for (i=0;i<5;i++)
    {
      wv[i] += corrv[i];
      wv[n-4+i]+=corrv[5-i];
    }
  
  double K0val=0.;
  for (i=0;i<=n;i++)
    K0val += h*wv[i]*exp(-a*tvec[i]-b/tvec[i])/tvec[i];

  return K0val;

}


double expint(int n, double x)
{
  int i,ii,nm1, MAXIT = 300;
  double a,b,c,d,del,fact,h,psi,ans=0;
  double FPMIN = 1.e-308;

  nm1=n-1;
  if (n < 0 || x < 0.0 || (fabs(x)<1e-18 && (n==0 || n==1)))
    {printf("bad arguments in expint %f\n",x);return -1;}
  else {
    if (n == 0) ans=exp(-x)/x;
    else {
      if (x == 0.0) ans=1.0/nm1;

      else {
	if (x > 1.0) {
	  b=x+n;
	  c=1.0/FPMIN;
	  d=1.0/b;
	  h=d;
	  for (i=1;i<=MAXIT;i++) {
	    a = -i*(nm1+i);
	    b += 2.0;
	    d=1.0/(a*d+b);
	    c=b+a/c;
	    del=c*d;
	    h *= del;
	    if (fabs(del-1.0) < 2.*EPS) {
	      ans=h*exp(-x);
	      return ans;
	    }
	  }
	  printf("continued fraction failed in expint");
	} else {
	  ans = (nm1!=0 ? 1.0/nm1 : -log(x)-EULER);
	  fact=1.0;
	  for (i=1;i<=MAXIT;i++) {
	    fact *= -x/i;
	    if (i != nm1) del = -fact/(i-nm1);
	    else {
	      psi = -EULER;
	      for (ii=1;ii<=nm1;ii++) psi += 1.0/ii;
	      del=fact*(-log(x)+psi);
	    }
	    ans += del;
	    if (fabs(del) < fabs(ans)*EPS) return ans;
	  }
	  printf("series failed in expint");
	}
      }
    }
  }
  return ans;
}


double polyval(double *p, double x)
{
  int i;
  double y=p[0];
  for (i=1;i<9;i++)
    {
      y = y*x + p[i];
    }
  return y;
}

double matlab_expint(double x)
{
  int j;
  double am1,am2,bm1,bm2,f,oldf,pterm,term,y,polyv,egamma,a,b,alpha,beta;
  double INF = 1.e308;
  double p[] = {-3.602693626336023e-09, -4.819538452140960e-07, -2.569498322115933e-05,
		-6.973790859534190e-04, -1.019573529845792e-02, -7.811863559248197e-02,
		-3.012432892762715e-01, -7.773807325735529e-01,  8.267661952366478e+00};

  polyv = polyval(p,x);
  if(polyv>=0)
    {
      egamma=0.57721566490153286061;
      y = -egamma-log(x);
      j = 1;
      pterm = x;
      term = x;
      
      while(fabs(term)>EPS)
	{
	  y += term;
	  j ++;
	  pterm = -x*pterm/j;
	  term  = pterm/j;
	}
    }
  else
    {
      am2 = 0.;bm2 = 1.; 
      am1 = 1.;bm1 = x;
      
      f    = am1 / bm1;
      oldf = INF;
      j    = 2;

      while( fabs(f-oldf) > 100.*EPS*fabs(f))
	{
	  alpha = j/2.;
	  
	  // calculate A(j), B(j), and f(j)
	  a = am1 + alpha * am2;
	  b = bm1 + alpha * bm2;
   
	  // save new normalized variables for next pass through the loop
	  //  note: normalization to avoid overflow or underflow
	  am2 = am1 / b;
	  bm2 = bm1 / b;
	  am1 = a   / b;
	  bm1 = 1;
	  
	  f = am1;
	  j++;
   
	  // calculate the coefficients for j odd
	  alpha = (j-1)/2.;
	  beta = x;
	  a = beta  * am1 + alpha * am2;
	  b = beta  * bm1 + alpha * bm2;
	  am2 = am1 / b;
	  bm2 = bm1 / b;
	  am1 = a   / b;
	  bm1 = 1;
	  oldf= f;
	  f   = am1;
	  j++;
	} // end of while loop
      
      y = exp(-x)*f;
    }


  return y;
}
