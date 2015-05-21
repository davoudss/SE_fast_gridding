# SE_fast_gridding: Fast Gaussian Gridding
The code is written in C and is accelerated using SSE and AVX. **_ _wrap_ _ ** and **_ _extend_ _** functions are written in **MATLAB** style but can be called in **C** style as well. **Please Do not change MATLAB style on master branch!**.

## How to run the sample code:
* N : (less than 200) number of charges e.g. 10.
* P : number of points in the support of Gaussians (11 for single and 21 for double precision).
* run the following commands on the command line

```sh
$ . setenv
$ ./a.out N P
```

### WARNING! using N > 100, the reference solution takes long time to run!

## to compute FORCE/POTENTIAL, or run with SSE/AVX, edit setenv file:
  * change **-DPOTENTIAL** to **-DFORCE** to compute force.
  * change **-DAVX** to **-DSSE**.
  * change **-DTHREE_PERIODIC** to **-DTWO_PERIODIC** for SE2P.

### NOTE: for SE2P, force calculation is not yet available.

### Contributers:
* Dag Lindbo
* Ludvig af Klinteberg
* Davoud Saffar Shamshirgar
