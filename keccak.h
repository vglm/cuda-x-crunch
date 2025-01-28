#pragma once

#include "types.hpp"
#include <chrono>
#include <iostream>

typedef union {
	uint8_t b[200];
	uint64_t q[25];
	uint32_t d[50];
} ethhash;

__device__ void sha3_keccakf(ethhash* const h);

#define rotate64(x, s) ((x << s) | (x >> (64U - s)))
#define rotate32(x, s) ((x << s) | (x >> (32U - s)))
#define bswap32(n) (rotate32(n & 0x00FF00FFU, 24U)|(rotate32(n, 8U) & 0x00FF00FFU))
__device__ mp_word mp_sub(mp_number * const r, const mp_number * const a, const mp_number * const b) ;

// Multiprecision subtraction of the modulus saved in mod. Underflow signalled via return value.
__device__ mp_word mp_sub_mod(mp_number * const r) ;


__device__ void mp_mod_sub(mp_number* const r, const mp_number* const a, const mp_number* const b);

__device__ void mp_mod_sub_const(mp_number* const r, const mp_number* const a, const mp_number* const b);


__device__ void mp_mod_sub_gx(mp_number* const r, const mp_number* const a) ;

// Multiprecision subtraction modulo M of G_y from a number.
// Specialization of mp_mod_sub in hope of performance gain.
__device__ void mp_mod_sub_gy(mp_number* const r, const mp_number* const a) ;

// Multiprecision addition. Overflow signalled via return value.
__device__ mp_word mp_add(mp_number* const r, const mp_number* const a) ;

// Multiprecision addition of the modulus saved in mod. Overflow signalled via return value.
__device__ mp_word mp_add_mod(mp_number* const r) ;

// Multiprecision addition of two numbers with one extra word each. Overflow signalled via return value.
__device__ mp_word mp_add_more(mp_number* const r, mp_word* const extraR, const mp_number* const a, const mp_word* const extraA) ;

// Multiprecision greater than or equal (>=) operator
__device__ mp_word mp_gte(const mp_number* const a, const mp_number* const b);

// Bit shifts a number with an extra word to the right one step
__device__ void mp_shr_extra(mp_number* const r, mp_word* const e);

// Bit shifts a number to the right one step
__device__ void mp_shr(mp_number* const r) ;

// Multiplies a number with a word and adds it to an existing number with an extra word, overflow of the extra word is signalled in return value
// This is a special function only used for modular multiplication
__device__ mp_word mp_mul_word_add_extra(mp_number* const r, const mp_number* const a, const mp_word w, mp_word* const extra) ;
// Multiplies a number with a word, potentially adds modhigher to it, and then subtracts it from en existing number, no extra words, no overflow
// This is a special function only used for modular multiplication
__device__ void mp_mul_mod_word_sub(mp_number* const r, const mp_word w, const bool withModHigher);

// Modular multiplication. Based on Algorithm 3 (and a series of hunches) from this article:
// https://www.esat.kuleuven.be/cosic/publications/article-1191.pdf
// When I first implemented it I never encountered a situation where the additional end steps
// of adding or subtracting the modulo was necessary. Maybe it's not for the particular modulo
// used in secp256k1, maybe the overflow bit can be skipped in to avoid 8 subtractions and
// trade it for the final steps? Maybe the final steps are necessary but seldom needed?
// I have no idea, for the time being I'll leave it like this, also see the comments at the
// beginning of this document under the title "Cutting corners".
__device__ void mp_mod_mul(mp_number* const r, const mp_number* const X, const mp_number* const Y) ;

// Modular inversion of a number.
__device__ void mp_mod_inverse(mp_number* const r) ;

// Elliptical point addition
// Does not handle points sharing X coordinate, this is a deliberate design choice.
// For more information on this choice see the beginning of this file.
__device__ void point_add(point* const r, point* const p, point* const o);


void test_keccakf();