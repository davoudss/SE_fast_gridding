# SE_fast_gridding: Fast Gaussian Gridding:

The code is written in C and is accelerated using SSE and AVX. "*_wrap_* and "*_extend_*" functions are written in
MATLAB style but can be called in C style as well. Please Do not change MATLAB style.

How to run the sample code:
Let N (less than 81000) be the number of charges and P is the 
number of points in the support of Gaussians
! NEVER use N > 100 as the reference solution takes long time to run

$ . setenv
$ ./a.out N P

** to compute FORCE/POTENTIAL, or run with SSE/AVX, edit setenv file:
  * change -DPOTENTIAL to -DFORCE to compute force.
  * change -DAVX to -DSSE
  * change -DTHREE_PERIODIUIC to -DTWO_PERIODIC for SE2P (see note)
NOTE: for SE2P, force calculation is not still available.

Contributers:
Dag Lindbo
Ludvig af Klinteberg
Davoud Saffar Shamshirgar
