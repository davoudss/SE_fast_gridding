#ifndef __MATH_X86_H_
#define __MATH_X86_H_

#include <immintrin.h>

#ifdef __AVX__
static inline __m256d
se_mm256_inv_pd(__m256d x)
{
  const __m256d two  = _mm256_set1_pd(2.0);

  /* Lookup instruction only exists in single precision, convert back and forth... */
  __m256d lu = _mm256_cvtps_pd(_mm_rcp_ps( _mm256_cvtpd_ps(x)));

  /* Perform two N-R steps for double precision */
  lu         = _mm256_mul_pd(lu, _mm256_sub_pd(two, _mm256_mul_pd(x, lu)));
  return _mm256_mul_pd(lu, _mm256_sub_pd(two, _mm256_mul_pd(x, lu)));
}

static inline __m256d
se_mm256_abs_pd(__m256d x)
{
  const __m256d signmask  = _mm256_castsi256_pd( _mm256_set_epi32(0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF,
								  0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF) );
  
  return _mm256_and_pd(x, signmask);
}

static __m256d
se_mm256_exp_pd(__m256d exparg)
{
  const __m256d argscale = _mm256_set1_pd(1.4426950408889634073599);
  /* Lower bound: We do not allow numbers that would lead to an IEEE fp representation exponent smaller than -126. */
  const __m256d arglimit = _mm256_set1_pd(1022.0);
  const __m128i expbase  = _mm_set1_epi32(1023);

  const __m256d invargscale0  = _mm256_set1_pd(6.93145751953125e-1);
  const __m256d invargscale1  = _mm256_set1_pd(1.42860682030941723212e-6);

  const __m256d P2       = _mm256_set1_pd(1.26177193074810590878e-4);
  const __m256d P1       = _mm256_set1_pd(3.02994407707441961300e-2);
  /* P0 == 1.0 */
  const __m256d Q3       = _mm256_set1_pd(3.00198505138664455042E-6);
  const __m256d Q2       = _mm256_set1_pd(2.52448340349684104192E-3);
  const __m256d Q1       = _mm256_set1_pd(2.27265548208155028766E-1);
  /* Q0 == 2.0 */
  const __m256d one      = _mm256_set1_pd(1.0);
  const __m256d two      = _mm256_set1_pd(2.0);

  __m256d       valuemask;
  __m256i       iexppart;
  __m128i       iexppart128a, iexppart128b;
  __m256d       fexppart;
  __m256d       intpart;
  __m256d       x, z, z2;
  __m256d       PolyP, PolyQ;

  x             = _mm256_mul_pd(exparg, argscale);

  iexppart128a  = _mm256_cvtpd_epi32(x);
  intpart       = _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT);

  /* Add exponent bias */
  iexppart128a   = _mm_add_epi32(iexppart128a, expbase);

  /* We now want to shift the exponent 52 positions left, but to achieve this we need
   * to separate the 128-bit register data into two registers (4x64-bit > 128bit)
   * shift them, and then merge into a single __m256d.
   * Elements 0/1 should end up in iexppart128a, and 2/3 in iexppart128b.
   * It doesnt matter what we put in the 2nd/4th position, since that data will be
   * shifted out and replaced with zeros.
   */
  iexppart128b   = _mm_shuffle_epi32(iexppart128a, _MM_SHUFFLE(3, 3, 2, 2));
  iexppart128a   = _mm_shuffle_epi32(iexppart128a, _MM_SHUFFLE(1, 1, 0, 0));

  iexppart128b   = _mm_slli_epi64(iexppart128b, 52);
  iexppart128a   = _mm_slli_epi64(iexppart128a, 52);

  iexppart  = _mm256_castsi128_si256(iexppart128a);
  iexppart  = _mm256_insertf128_si256(iexppart, iexppart128b, 0x1);

  valuemask = _mm256_cmp_pd(arglimit, se_mm256_abs_pd(x), _CMP_GE_OQ);
  fexppart  = _mm256_and_pd(valuemask, _mm256_castsi256_pd(iexppart));

  z         = _mm256_sub_pd(exparg, _mm256_mul_pd(invargscale0, intpart));
  z         = _mm256_sub_pd(z, _mm256_mul_pd(invargscale1, intpart));

  z2        = _mm256_mul_pd(z, z);

  PolyQ     = _mm256_mul_pd(Q3, z2);
  PolyQ     = _mm256_add_pd(PolyQ, Q2);
  PolyP     = _mm256_mul_pd(P2, z2);
  PolyQ     = _mm256_mul_pd(PolyQ, z2);
  PolyP     = _mm256_add_pd(PolyP, P1);
  PolyQ     = _mm256_add_pd(PolyQ, Q1);
  PolyP     = _mm256_mul_pd(PolyP, z2);
  PolyQ     = _mm256_mul_pd(PolyQ, z2);
  PolyP     = _mm256_add_pd(PolyP, one);
  PolyQ     = _mm256_add_pd(PolyQ, two);

  PolyP     = _mm256_mul_pd(PolyP, z);

  z         = _mm256_mul_pd(PolyP, se_mm256_inv_pd(_mm256_sub_pd(PolyQ, PolyP)));
  z         = _mm256_add_pd(one, _mm256_mul_pd(two, z));

  z         = _mm256_mul_pd(z, fexppart);

  return z;
}

#elif __SSE4_2__
static inline __m128d
se_mm_inv_pd(__m128d x)
{
  const __m128d two  = _mm_set1_pd(2.0);

  // Lookup instruction only exists in single precision, convert back and forth... //
  __m128d lu = _mm_cvtps_pd(_mm_rcp_ps( _mm_cvtpd_ps(x)));

  // Perform two N-R steps for double precision //
  lu         = _mm_mul_pd(lu, _mm_sub_pd(two, _mm_mul_pd(x, lu)));
  return _mm_mul_pd(lu, _mm_sub_pd(two, _mm_mul_pd(x, lu)));
}

static inline __m128d
se_mm_abs_pd(__m128d x)
{
  const __m128d signmask  = _mm_castsi128_pd( _mm_set_epi32(0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF) );

  return _mm_and_pd(x, signmask);
}

static __m128d
se_mm_exp_pd(__m128d exparg)
{
  const __m128d argscale = _mm_set1_pd(1.4426950408889634073599);
  /* Lower bound: We do not allow numbers that would lead to an IEEE fp representation exponent smaller than -126. */
  const __m128d arglimit = _mm_set1_pd(1022.0);
  const __m128i expbase  = _mm_set1_epi32(1023);

  const __m128d invargscale0  = _mm_set1_pd(6.93145751953125e-1);
  const __m128d invargscale1  = _mm_set1_pd(1.42860682030941723212e-6);

  const __m128d P2       = _mm_set1_pd(1.26177193074810590878e-4);
  const __m128d P1       = _mm_set1_pd(3.02994407707441961300e-2);
  /* P0 == 1.0 */
  const __m128d Q3       = _mm_set1_pd(3.00198505138664455042E-6);
  const __m128d Q2       = _mm_set1_pd(2.52448340349684104192E-3);
  const __m128d Q1       = _mm_set1_pd(2.27265548208155028766E-1);
  /* Q0 == 2.0 */
  const __m128d one      = _mm_set1_pd(1.0);
  const __m128d two      = _mm_set1_pd(2.0);

  __m128d       valuemask;
  __m128i       iexppart;
  __m128d       fexppart;
  __m128d       intpart;
  __m128d       x, z, z2;
  __m128d       PolyP, PolyQ;

  x             = _mm_mul_pd(exparg, argscale);

  iexppart  = _mm_cvtpd_epi32(x);
  intpart   = _mm_round_pd(x, _MM_FROUND_TO_NEAREST_INT);

  /* The two lowest elements of iexppart now contains 32-bit numbers with a correctly biased exponent.
   * To be able to shift it into the exponent for a double precision number we first need to
   * shuffle so that the lower half contains the first element, and the upper half the second.
   * This should really be done as a zero-extension, but since the next instructions will shift
   * the registers left by 52 bits it doesn't matter what we put there - it will be shifted out.
   * (thus we just use element 2 from iexppart).
   */
  iexppart  = _mm_shuffle_epi32(iexppart, _MM_SHUFFLE(2, 1, 2, 0));

  /* Do the shift operation on the 64-bit registers */
  iexppart  = _mm_add_epi32(iexppart, expbase);
  iexppart  = _mm_slli_epi64(iexppart, 52);

  valuemask = _mm_cmpge_pd(arglimit, se_mm_abs_pd(x));
  fexppart  = _mm_and_pd(valuemask, _mm_castsi128_pd(iexppart));

  z         = _mm_sub_pd(exparg, _mm_mul_pd(invargscale0, intpart));
  z         = _mm_sub_pd(z, _mm_mul_pd(invargscale1, intpart));

  z2        = _mm_mul_pd(z, z);

  PolyQ     = _mm_mul_pd(Q3, z2);
  PolyQ     = _mm_add_pd(PolyQ, Q2);
  PolyP     = _mm_mul_pd(P2, z2);
  PolyQ     = _mm_mul_pd(PolyQ, z2);
  PolyP     = _mm_add_pd(PolyP, P1);
  PolyQ     = _mm_add_pd(PolyQ, Q1);
  PolyP     = _mm_mul_pd(PolyP, z2);
  PolyQ     = _mm_mul_pd(PolyQ, z2);
  PolyP     = _mm_add_pd(PolyP, one);
  PolyQ     = _mm_add_pd(PolyQ, two);

  PolyP     = _mm_mul_pd(PolyP, z);

  z         = _mm_mul_pd(PolyP, se_mm_inv_pd(_mm_sub_pd(PolyQ, PolyP)));
  z         = _mm_add_pd(one, _mm_mul_pd(two, z));

  z         = _mm_mul_pd(z, fexppart);

  return z;
}
#endif // AVX

#endif // _MATH_X86_H_
