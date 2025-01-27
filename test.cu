﻿/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "help.hpp"
#include "ArgParser.hpp"
#include "particle.h"
#include <stdlib.h>
#include <stdio.h>
#include "precomp.hpp"
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

typedef union {
	uint8_t b[200];
	uint64_t q[25];
	uint32_t d[50];
} ethhash;

#define rotate(x, s) ((x << s) | (x >> (64 - s)))
#define bswap32(n) (rotate(n & 0x00FF00FF, 24U)|(rotate(n, 8U) & 0x00FF00FF))



#define mul_hi(a, b) __umulhi(a, b)

// mod              = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
__device__ const mp_number mod              = { {0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

// tripleNegativeGx = 0x92c4cc831269ccfaff1ed83e946adeeaf82c096e76958573f2287becbb17b196
__device__ const mp_number tripleNegativeGx = { {0xbb17b196, 0xf2287bec, 0x76958573, 0xf82c096e, 0x946adeea, 0xff1ed83e, 0x1269ccfa, 0x92c4cc83 } };

// doubleNegativeGy = 0x6f8a4b11b2b8773544b60807e3ddeeae05d0976eb2f557ccc7705edf09de52bf
__device__ const mp_number doubleNegativeGy = { {0x09de52bf, 0xc7705edf, 0xb2f557cc, 0x05d0976e, 0xe3ddeeae, 0x44b60807, 0xb2b87735, 0x6f8a4b11} };

// negativeGy       = 0xb7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777
__device__ const mp_number negativeGy       = { {0x04ef2777, 0x63b82f6f, 0x597aabe6, 0x02e84bb7, 0xf1eef757, 0xa25b0403, 0xd95c3b9a, 0xb7c52588 } };

// Multiprecision subtraction. Underflow signalled via return value.
__device__ mp_word mp_sub(mp_number * const r, const mp_number * const a, const mp_number * const b) {
	mp_word t, c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		t = a->d[i] - b->d[i] - c;
		c = t > a->d[i] ? 1 : (t == a->d[i] ? c : 0);

		r->d[i] = t;
	}

	return c;
}



// Multiprecision subtraction of the modulus saved in mod. Underflow signalled via return value.
__device__ mp_word mp_sub_mod(mp_number * const r) {
	mp_number mod = { {0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

	mp_word t, c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		t = r->d[i] - mod.d[i] - c;
		c = t > r->d[i] ? 1 : (t == r->d[i] ? c : 0);

		r->d[i] = t;
	}

	return c;
}


__device__ void mp_mod_sub(mp_number* const r, const mp_number* const a, const mp_number* const b) {
	mp_word i, t, c = 0;

	for (i = 0; i < MP_WORDS; ++i) {
		t = a->d[i] - b->d[i] - c;
		c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1);

		r->d[i] = t;
	}

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

__device__ void mp_mod_sub_const(mp_number* const r, const mp_number* const a, const mp_number* const b) {
	mp_word i, t, c = 0;

	for (i = 0; i < MP_WORDS; ++i) {
		t = a->d[i] - b->d[i] - c;
		c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1);

		r->d[i] = t;
	}

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}


__device__ void mp_mod_sub_gx(mp_number* const r, const mp_number* const a) {
	mp_word i, t, c = 0;

	t = a->d[0] - 0x16f81798; c = t < a->d[0] ? 0 : (t == a->d[0] ? c : 1); r->d[0] = t;
	t = a->d[1] - 0x59f2815b - c; c = t < a->d[1] ? 0 : (t == a->d[1] ? c : 1); r->d[1] = t;
	t = a->d[2] - 0x2dce28d9 - c; c = t < a->d[2] ? 0 : (t == a->d[2] ? c : 1); r->d[2] = t;
	t = a->d[3] - 0x029bfcdb - c; c = t < a->d[3] ? 0 : (t == a->d[3] ? c : 1); r->d[3] = t;
	t = a->d[4] - 0xce870b07 - c; c = t < a->d[4] ? 0 : (t == a->d[4] ? c : 1); r->d[4] = t;
	t = a->d[5] - 0x55a06295 - c; c = t < a->d[5] ? 0 : (t == a->d[5] ? c : 1); r->d[5] = t;
	t = a->d[6] - 0xf9dcbbac - c; c = t < a->d[6] ? 0 : (t == a->d[6] ? c : 1); r->d[6] = t;
	t = a->d[7] - 0x79be667e - c; c = t < a->d[7] ? 0 : (t == a->d[7] ? c : 1); r->d[7] = t;

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

// Multiprecision subtraction modulo M of G_y from a number.
// Specialization of mp_mod_sub in hope of performance gain.
__device__ void mp_mod_sub_gy(mp_number* const r, const mp_number* const a) {
	mp_word i, t, c = 0;

	t = a->d[0] - 0xfb10d4b8; c = t < a->d[0] ? 0 : (t == a->d[0] ? c : 1); r->d[0] = t;
	t = a->d[1] - 0x9c47d08f - c; c = t < a->d[1] ? 0 : (t == a->d[1] ? c : 1); r->d[1] = t;
	t = a->d[2] - 0xa6855419 - c; c = t < a->d[2] ? 0 : (t == a->d[2] ? c : 1); r->d[2] = t;
	t = a->d[3] - 0xfd17b448 - c; c = t < a->d[3] ? 0 : (t == a->d[3] ? c : 1); r->d[3] = t;
	t = a->d[4] - 0x0e1108a8 - c; c = t < a->d[4] ? 0 : (t == a->d[4] ? c : 1); r->d[4] = t;
	t = a->d[5] - 0x5da4fbfc - c; c = t < a->d[5] ? 0 : (t == a->d[5] ? c : 1); r->d[5] = t;
	t = a->d[6] - 0x26a3c465 - c; c = t < a->d[6] ? 0 : (t == a->d[6] ? c : 1); r->d[6] = t;
	t = a->d[7] - 0x483ada77 - c; c = t < a->d[7] ? 0 : (t == a->d[7] ? c : 1); r->d[7] = t;

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

// Multiprecision addition. Overflow signalled via return value.
__device__ mp_word mp_add(mp_number* const r, const mp_number* const a) {
	mp_word c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		r->d[i] += a->d[i] + c;
		c = r->d[i] < a->d[i] ? 1 : (r->d[i] == a->d[i] ? c : 0);
	}

	return c;
}

// Multiprecision addition of the modulus saved in mod. Overflow signalled via return value.
__device__ mp_word mp_add_mod(mp_number* const r) {
	mp_word c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		r->d[i] += mod.d[i] + c;
		c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
	}

	return c;
}

// Multiprecision addition of two numbers with one extra word each. Overflow signalled via return value.
__device__ mp_word mp_add_more(mp_number* const r, mp_word* const extraR, const mp_number* const a, const mp_word* const extraA) {
	const mp_word c = mp_add(r, a);
	*extraR += *extraA + c;
	return *extraR < *extraA ? 1 : (*extraR == *extraA ? c : 0);
}

// Multiprecision greater than or equal (>=) operator
__device__ mp_word mp_gte(const mp_number* const a, const mp_number* const b) {
	mp_word l = 0, g = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		if (a->d[i] < b->d[i]) l |= (1 << i);
		if (a->d[i] > b->d[i]) g |= (1 << i);
	}

	return g >= l;
}

// Bit shifts a number with an extra word to the right one step
__device__ void mp_shr_extra(mp_number* const r, mp_word* const e) {
	r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
	r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
	r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
	r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
	r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
	r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
	r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
	r->d[7] = (*e << 31) | (r->d[7] >> 1);
	*e >>= 1;
}

// Bit shifts a number to the right one step
__device__ void mp_shr(mp_number* const r) {
	r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
	r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
	r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
	r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
	r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
	r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
	r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
	r->d[7] >>= 1;
}

// Multiplies a number with a word and adds it to an existing number with an extra word, overflow of the extra word is signalled in return value
// This is a special function only used for modular multiplication
__device__ mp_word mp_mul_word_add_extra(mp_number* const r, const mp_number* const a, const mp_word w, mp_word* const extra) {
	mp_word cM = 0; // Carry for multiplication
	mp_word cA = 0; // Carry for addition
	mp_word tM = 0; // Temporary storage for multiplication

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		tM = (a->d[i] * w + cM);
		cM = mul_hi(a->d[i], w) + (tM < cM);

		r->d[i] += tM + cA;
		cA = r->d[i] < tM ? 1 : (r->d[i] == tM ? cA : 0);
	}

	*extra += cM + cA;
	return *extra < cM ? 1 : (*extra == cM ? cA : 0);
}

// Multiplies a number with a word, potentially adds modhigher to it, and then subtracts it from en existing number, no extra words, no overflow
// This is a special function only used for modular multiplication
__device__ void mp_mul_mod_word_sub(mp_number* const r, const mp_word w, const bool withModHigher) {
	// Having these numbers declared here instead of using the global values in __constant address space seems to lead
	// to better optimizations by the compiler on my GTX 1070.
	mp_number mod = { { 0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };
	mp_number modhigher = { {0x00000000, 0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

	mp_word cM = 0; // Carry for multiplication
	mp_word cS = 0; // Carry for subtraction
	mp_word tS = 0; // Temporary storage for subtraction
	mp_word tM = 0; // Temporary storage for multiplication
	mp_word cA = 0; // Carry for addition of modhigher

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		tM = (mod.d[i] * w + cM);
		cM = mul_hi(mod.d[i], w) + (tM < cM);

		tM += (withModHigher ? modhigher.d[i] : 0) + cA;
		cA = tM < (withModHigher ? modhigher.d[i] : 0) ? 1 : (tM == (withModHigher ? modhigher.d[i] : 0) ? cA : 0);

		tS = r->d[i] - tM - cS;
		cS = tS > r->d[i] ? 1 : (tS == r->d[i] ? cS : 0);

		r->d[i] = tS;
	}
}

// Modular multiplication. Based on Algorithm 3 (and a series of hunches) from this article:
// https://www.esat.kuleuven.be/cosic/publications/article-1191.pdf
// When I first implemented it I never encountered a situation where the additional end steps
// of adding or subtracting the modulo was necessary. Maybe it's not for the particular modulo
// used in secp256k1, maybe the overflow bit can be skipped in to avoid 8 subtractions and
// trade it for the final steps? Maybe the final steps are necessary but seldom needed?
// I have no idea, for the time being I'll leave it like this, also see the comments at the
// beginning of this document under the title "Cutting corners".
__device__ void mp_mod_mul(mp_number* const r, const mp_number* const X, const mp_number* const Y) {
	mp_number Z = { {0} };
	mp_word extraWord;

	for (int i = MP_WORDS - 1; i >= 0; --i) {
		// Z = Z * 2^32
		extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;

		// Z = Z + X * Y_i
		bool overflow = mp_mul_word_add_extra(&Z, X, Y->d[i], &extraWord);

		// Z = Z - qM
		mp_mul_mod_word_sub(&Z, extraWord, overflow);
	}

	*r = Z;
}

// Modular inversion of a number. 
__device__ void mp_mod_inverse(mp_number* const r) {
	mp_number A = { { 1 } };
	mp_number C = { { 0 } };
	mp_number v = mod;

	mp_word extraA = 0;
	mp_word extraC = 0;

	while (r->d[0] || r->d[1] || r->d[2] || r->d[3] || r->d[4] || r->d[5] || r->d[6] || r->d[7]) {
		while (!(r->d[0] & 1)) {
			mp_shr(r);
			if (A.d[0] & 1) {
				extraA += mp_add_mod(&A);
			}

			mp_shr_extra(&A, &extraA);
		}

		while (!(v.d[0] & 1)) {
			mp_shr(&v);
			if (C.d[0] & 1) {
				extraC += mp_add_mod(&C);
			}

			mp_shr_extra(&C, &extraC);
		}

		if (mp_gte(r, &v)) {
			mp_sub(r, r, &v);
			mp_add_more(&A, &extraA, &C, &extraC);
		}
		else {
			mp_sub(&v, &v, r);
			mp_add_more(&C, &extraC, &A, &extraA);
		}
	}

	while (extraC) {
		extraC -= mp_sub_mod(&C);
	}

	v = mod;
	mp_sub(r, &v, &C);
}


// Elliptical point addition
// Does not handle points sharing X coordinate, this is a deliberate design choice.
// For more information on this choice see the beginning of this file.
__device__ void point_add(point* const r, point* const p, point* const o) {
	mp_number tmp;
	mp_number newX;
	mp_number newY;

	mp_mod_sub(&tmp, &o->x, &p->x);

	mp_mod_inverse(&tmp);

	mp_mod_sub(&newX, &o->y, &p->y);
	mp_mod_mul(&tmp, &tmp, &newX);

	mp_mod_mul(&newX, &tmp, &tmp);
	mp_mod_sub(&newX, &newX, &p->x);
	mp_mod_sub(&newX, &newX, &o->x);

	mp_mod_sub(&newY, &p->x, &newX);
	mp_mod_mul(&newY, &newY, &tmp);
	mp_mod_sub(&newY, &newY, &p->y);

	r->x = newX;
	r->y = newY;
}




/* ------------------------------------------------------------------------ */
/* Profanity.                                                               */
/* ------------------------------------------------------------------------ */
typedef struct {
	uint32_t found;
	uint32_t foundId;
	uint8_t foundHash[20];
} result;

__device__ void profanity_init_seed(const point* const precomp, point* const p, bool* const pIsFirst, const size_t precompOffset, const uint64_t seed) {
	point o;

	for (uint8_t i = 0; i < 8; ++i) {
		const uint8_t shift = i * 8;
		const uint8_t byte = (seed >> shift) & 0xFF;

		if (byte) {
			o = precomp[precompOffset + i * 255 + byte - 1];
			if (*pIsFirst) {
				*p = o;
				*pIsFirst = false;
			}
			else {
				point_add(p, p, &o);
			}
		}
	}
}

#define PROFANITY_INVERSE_SIZE 255


__device__ void profanity_init(int i, const point* const precomp, mp_number* const pDeltaX, mp_number* const pPrevLambda, result* const pResult, const ulong4 seed, const ulong4 seedX, const ulong4 seedY) {
	const size_t id = (threadIdx.x + blockIdx.x * blockDim.x) * PROFANITY_INVERSE_SIZE + i;
	
	/*
	point p = {
		.x = {.d = {
			seedX.x & 0xFFFFFFFF, seedX.x >> 32,
			seedX.y & 0xFFFFFFFF, seedX.y >> 32,
			seedX.z & 0xFFFFFFFF, seedX.z >> 32,
			seedX.w & 0xFFFFFFFF, seedX.w >> 32,
		}},
		.y = {.d = {
			seedY.x & 0xFFFFFFFF, seedY.x >> 32,
			seedY.y & 0xFFFFFFFF, seedY.y >> 32,
			seedY.z & 0xFFFFFFFF, seedY.z >> 32,
			seedY.w & 0xFFFFFFFF, seedY.w >> 32,
		}},
	};*/

	point p;
	p.x.d[0] = seedX.x & 0xFFFFFFFF;
	p.x.d[1] = seedX.x >> 32;
	p.x.d[2] = seedX.y & 0xFFFFFFFF;
	p.x.d[3] = seedX.y >> 32;
	p.x.d[4] = seedX.z & 0xFFFFFFFF;
	p.x.d[5] = seedX.z >> 32;
	p.x.d[6] = seedX.w & 0xFFFFFFFF;
	p.x.d[7] = seedX.w >> 32;
	p.y.d[0] = seedY.x & 0xFFFFFFFF;
	p.y.d[1] = seedY.x >> 32;
	p.y.d[2] = seedY.y & 0xFFFFFFFF;
	p.y.d[3] = seedY.y >> 32;
	p.y.d[4] = seedY.z & 0xFFFFFFFF;
	p.y.d[5] = seedY.z >> 32;
	p.y.d[6] = seedY.w & 0xFFFFFFFF;
	p.y.d[7] = seedY.w >> 32;

	point p_random;
	bool bIsFirst = true;

	mp_number tmp1, tmp2;
	point tmp3;

	// Calculate k*G where k = seed.wzyx (in other words, find the point indicated by the private key represented in seed)
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 0, seed.x);
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 1, seed.y);
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 2, seed.z);
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 3, seed.w + id);
	point_add(&p, &p, &p_random);

	// Calculate current lambda in this point
	mp_mod_sub_gx(&tmp1, &p.x);
	mp_mod_inverse(&tmp1);

	mp_mod_sub_gy(&tmp2, &p.y);
	mp_mod_mul(&tmp1, &tmp1, &tmp2);

	// Jump to next point (precomp[0] is the generator point G)
	tmp3 = precomp[0];
	point_add(&p, &tmp3, &p);

	// pDeltaX should contain the delta (x - G_x)
	mp_mod_sub_gx(&p.x, &p.x);

	pDeltaX[id] = p.x;
	pPrevLambda[id] = tmp1;

	for (uint8_t i = 0; i < 40 + 1; ++i) {
		pResult[i].found = 0;
	}
}


// This kernel calculates several modular inversions at once with just one inverse.
// It's an implementation of Algorithm 2.11 from Modern Computer Arithmetic:
// https://members.loria.fr/PZimmermann/mca/pub226.html 
//
// My RX 480 is very sensitive to changes in the second loop and sometimes I have
// to make seemingly non-functional changes to the code to make the compiler
// generate the most optimized version.
__device__ void profanity_inverse(const mp_number* const pDeltaX, mp_number* const pInverse) {
	const size_t id = (threadIdx.x + blockIdx.x * blockDim.x) * PROFANITY_INVERSE_SIZE;

	// negativeDoubleGy = 0x6f8a4b11b2b8773544b60807e3ddeeae05d0976eb2f557ccc7705edf09de52bf
	mp_number negativeDoubleGy = { {0x09de52bf, 0xc7705edf, 0xb2f557cc, 0x05d0976e, 0xe3ddeeae, 0x44b60807, 0xb2b87735, 0x6f8a4b11 } };

	mp_number copy1, copy2;
	mp_number buffer[PROFANITY_INVERSE_SIZE];
	mp_number buffer2[PROFANITY_INVERSE_SIZE];

	// We initialize buffer and buffer2 such that:
	// buffer[i] = pDeltaX[id] * pDeltaX[id + 1] * pDeltaX[id + 2] * ... * pDeltaX[id + i]
	// buffer2[i] = pDeltaX[id + i]
	buffer[0] = pDeltaX[id];
	for (uint32_t i = 1; i < PROFANITY_INVERSE_SIZE; ++i) {
		buffer2[i] = pDeltaX[id + i];
		mp_mod_mul(&buffer[i], &buffer2[i], &buffer[i - 1]);
	}

	// Take the inverse of all x-values combined
	copy1 = buffer[PROFANITY_INVERSE_SIZE - 1];
	mp_mod_inverse(&copy1);

	// We multiply in -2G_y together with the inverse so that we have:
	//            - 2 * G_y
	//  ----------------------------
	//  x_0 * x_1 * x_2 * x_3 * ...
	mp_mod_mul(&copy1, &copy1, &negativeDoubleGy);

	// Multiply out each individual inverse using the buffers
	for (uint32_t i = PROFANITY_INVERSE_SIZE - 1; i > 0; --i) {
		mp_mod_mul(&copy2, &copy1, &buffer[i - 1]);
		mp_mod_mul(&copy1, &copy1, &buffer2[i]);
		pInverse[id + i] = copy2;
	}

	pInverse[id] = copy1;
}

__device__ void sha3_keccakf(ethhash* const h);

__device__ void profanity_iterate(mp_number* const pDeltaX, mp_number* const pInverse, mp_number* const pPrevLambda) {
	const size_t id = (threadIdx.x + blockIdx.x * blockDim.x) * PROFANITY_INVERSE_SIZE;

	// negativeGx = 0x8641998106234453aa5f9d6a3178f4f8fd640324d231d726a60d7ea3e907e497
	mp_number negativeGx = { {0xe907e497, 0xa60d7ea3, 0xd231d726, 0xfd640324, 0x3178f4f8, 0xaa5f9d6a, 0x06234453, 0x86419981 } };

	ethhash h = { { 0 } };

	mp_number dX = pDeltaX[id];
	mp_number tmp = pInverse[id];
	mp_number lambda = pPrevLambda[id];

	// λ' = - (2G_y) / d' - λ <=> lambda := pInversedNegativeDoubleGy[id] - pPrevLambda[id]
	mp_mod_sub(&lambda, &tmp, &lambda);

	// λ² = λ * λ <=> tmp := lambda * lambda = λ²
	mp_mod_mul(&tmp, &lambda, &lambda);

	// d' = λ² - d - 3g = (-3g) - (d - λ²) <=> x := tripleNegativeGx - (x - tmp)
	mp_mod_sub(&dX, &dX, &tmp);
	mp_mod_sub_const(&dX, &tripleNegativeGx, &dX);

	pDeltaX[id] = dX;
	pPrevLambda[id] = lambda;

	// Calculate y from dX and lambda
	// y' = (-G_Y) - λ * d' <=> p.y := negativeGy - (p.y * p.x)
	mp_mod_mul(&tmp, &lambda, &dX);
	mp_mod_sub_const(&tmp, &negativeGy, &tmp);

	// Restore X coordinate from delta value
	mp_mod_sub(&dX, &dX, &negativeGx);

	// Initialize Keccak structure with point coordinates in big endian
	h.d[0] = bswap32(dX.d[MP_WORDS - 1]);
	h.d[1] = bswap32(dX.d[MP_WORDS - 2]);
	h.d[2] = bswap32(dX.d[MP_WORDS - 3]);
	h.d[3] = bswap32(dX.d[MP_WORDS - 4]);
	h.d[4] = bswap32(dX.d[MP_WORDS - 5]);
	h.d[5] = bswap32(dX.d[MP_WORDS - 6]);
	h.d[6] = bswap32(dX.d[MP_WORDS - 7]);
	h.d[7] = bswap32(dX.d[MP_WORDS - 8]);
	h.d[8] = bswap32(tmp.d[MP_WORDS - 1]);
	h.d[9] = bswap32(tmp.d[MP_WORDS - 2]);
	h.d[10] = bswap32(tmp.d[MP_WORDS - 3]);
	h.d[11] = bswap32(tmp.d[MP_WORDS - 4]);
	h.d[12] = bswap32(tmp.d[MP_WORDS - 5]);
	h.d[13] = bswap32(tmp.d[MP_WORDS - 6]);
	h.d[14] = bswap32(tmp.d[MP_WORDS - 7]);
	h.d[15] = bswap32(tmp.d[MP_WORDS - 8]);
	h.d[16] ^= 0x01; // length 64

	sha3_keccakf(&h);

	// Save public address hash in pInverse, only used as interim storage until next cycle
	pInverse[id].d[0] = h.d[3];
	pInverse[id].d[1] = h.d[4];
	pInverse[id].d[2] = h.d[5];
	pInverse[id].d[3] = h.d[6];
	pInverse[id].d[4] = h.d[7];
}


#define TH_ELT(t, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4) \
{ \
    t = rotate((uint64_t)(d0 ^ d1 ^ d2 ^ d3 ^ d4), (uint64_t)1) ^ (c0 ^ c1 ^ c2 ^ c3 ^ c4); \
}

#define THETA(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
    TH_ELT(t0, s40, s41, s42, s43, s44, s10, s11, s12, s13, s14); \
    TH_ELT(t1, s00, s01, s02, s03, s04, s20, s21, s22, s23, s24); \
    TH_ELT(t2, s10, s11, s12, s13, s14, s30, s31, s32, s33, s34); \
    TH_ELT(t3, s20, s21, s22, s23, s24, s40, s41, s42, s43, s44); \
    TH_ELT(t4, s30, s31, s32, s33, s34, s00, s01, s02, s03, s04); \
    s00 ^= t0; s01 ^= t0; s02 ^= t0; s03 ^= t0; s04 ^= t0; \
    s10 ^= t1; s11 ^= t1; s12 ^= t1; s13 ^= t1; s14 ^= t1; \
    s20 ^= t2; s21 ^= t2; s22 ^= t2; s23 ^= t2; s24 ^= t2; \
    s30 ^= t3; s31 ^= t3; s32 ^= t3; s33 ^= t3; s34 ^= t3; \
    s40 ^= t4; s41 ^= t4; s42 ^= t4; s43 ^= t4; s44 ^= t4; \
}

#define RHOPI(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
	t0  = rotate(s10, (uint64_t) 1);  \
	s10 = rotate(s11, (uint64_t)44); \
	s11 = rotate(s41, (uint64_t)20); \
	s41 = rotate(s24, (uint64_t)61); \
	s24 = rotate(s42, (uint64_t)39); \
	s42 = rotate(s04, (uint64_t)18); \
	s04 = rotate(s20, (uint64_t)62); \
	s20 = rotate(s22, (uint64_t)43); \
	s22 = rotate(s32, (uint64_t)25); \
	s32 = rotate(s43, (uint64_t) 8); \
	s43 = rotate(s34, (uint64_t)56); \
	s34 = rotate(s03, (uint64_t)41); \
	s03 = rotate(s40, (uint64_t)27); \
	s40 = rotate(s44, (uint64_t)14); \
	s44 = rotate(s14, (uint64_t) 2); \
	s14 = rotate(s31, (uint64_t)55); \
	s31 = rotate(s13, (uint64_t)45); \
	s13 = rotate(s01, (uint64_t)36); \
	s01 = rotate(s30, (uint64_t)28); \
	s30 = rotate(s33, (uint64_t)21); \
	s33 = rotate(s23, (uint64_t)15); \
	s23 = rotate(s12, (uint64_t)10); \
	s12 = rotate(s21, (uint64_t) 6); \
	s21 = rotate(s02, (uint64_t) 3); \
	s02 = t0; \
}

#define KHI(s00, s01, s02, s03, s04, \
            s10, s11, s12, s13, s14, \
            s20, s21, s22, s23, s24, \
            s30, s31, s32, s33, s34, \
            s40, s41, s42, s43, s44) \
{ \
    t0 = s00 ^ (~s10 &  s20); \
    t1 = s10 ^ (~s20 &  s30); \
    t2 = s20 ^ (~s30 &  s40); \
    t3 = s30 ^ (~s40 &  s00); \
    t4 = s40 ^ (~s00 &  s10); \
    s00 = t0; s10 = t1; s20 = t2; s30 = t3; s40 = t4; \
    \
    t0 = s01 ^ (~s11 &  s21); \
    t1 = s11 ^ (~s21 &  s31); \
    t2 = s21 ^ (~s31 &  s41); \
    t3 = s31 ^ (~s41 &  s01); \
    t4 = s41 ^ (~s01 &  s11); \
    s01 = t0; s11 = t1; s21 = t2; s31 = t3; s41 = t4; \
    \
    t0 = s02 ^ (~s12 &  s22); \
    t1 = s12 ^ (~s22 &  s32); \
    t2 = s22 ^ (~s32 &  s42); \
    t3 = s32 ^ (~s42 &  s02); \
    t4 = s42 ^ (~s02 &  s12); \
    s02 = t0; s12 = t1; s22 = t2; s32 = t3; s42 = t4; \
    \
    t0 = s03 ^ (~s13 &  s23); \
    t1 = s13 ^ (~s23 &  s33); \
    t2 = s23 ^ (~s33 &  s43); \
    t3 = s33 ^ (~s43 &  s03); \
    t4 = s43 ^ (~s03 &  s13); \
    s03 = t0; s13 = t1; s23 = t2; s33 = t3; s43 = t4; \
    \
    t0 = s04 ^ (~s14 &  s24); \
    t1 = s14 ^ (~s24 &  s34); \
    t2 = s24 ^ (~s34 &  s44); \
    t3 = s34 ^ (~s44 &  s04); \
    t4 = s44 ^ (~s04 &  s14); \
    s04 = t0; s14 = t1; s24 = t2; s34 = t3; s44 = t4; \
}

#define IOTA(s00, r) { s00 ^= r; }

__device__ uint64_t keccakf_rndc[24] = {
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// Barely a bottleneck. No need to tinker more.
__device__ void sha3_keccakf(ethhash* const h)
{
	h->d[33] ^= 0x80000000;
	uint64_t t0, t1, t2, t3, t4;

	// Unrolling and removing PI stage gave negligable performance on GTX 1070.
	for (int i = 0; i < 24; ++i) {
		THETA(h->q[0], h->q[5], h->q[10], h->q[15], h->q[20], h->q[1], h->q[6], h->q[11], h->q[16], h->q[21], h->q[2], h->q[7], h->q[12], h->q[17], h->q[22], h->q[3], h->q[8], h->q[13], h->q[18], h->q[23], h->q[4], h->q[9], h->q[14], h->q[19], h->q[24]);
		RHOPI(h->q[0], h->q[5], h->q[10], h->q[15], h->q[20], h->q[1], h->q[6], h->q[11], h->q[16], h->q[21], h->q[2], h->q[7], h->q[12], h->q[17], h->q[22], h->q[3], h->q[8], h->q[13], h->q[18], h->q[23], h->q[4], h->q[9], h->q[14], h->q[19], h->q[24]);
		KHI(h->q[0], h->q[5], h->q[10], h->q[15], h->q[20], h->q[1], h->q[6], h->q[11], h->q[16], h->q[21], h->q[2], h->q[7], h->q[12], h->q[17], h->q[22], h->q[3], h->q[8], h->q[13], h->q[18], h->q[23], h->q[4], h->q[9], h->q[14], h->q[19], h->q[24]);
		IOTA(h->q[0], keccakf_rndc[i]);
	}
}


__global__ void advanceParticles(float dt, particle * pArray, int nParticles, point* precomp, mp_number* pointsDeltaX, mp_number* pPrevLambda, mp_number* pInverse)
{
	ulong4 seed;
	seed.x = 0x0;
	seed.y = 0x0;
	seed.z = 0x0;
	seed.w = 0;



	result pResult = { 0 };

	for (int i = 0; i < PROFANITY_INVERSE_SIZE; i++)
	{
		profanity_init(i, precomp, pointsDeltaX, pPrevLambda, &pResult, seed, seed, seed);
	}

	profanity_inverse(pointsDeltaX, pInverse);
	profanity_iterate(pointsDeltaX, pInverse, pPrevLambda);
	//pInverse[(threadIdx.x + blockIdx.x * blockDim.x) * PROFANITY_INVERSE_SIZE].d[0] = 222222222;
}


static std::string toHex(const uint8_t* const s, const size_t len) {
	std::string b("0123456789abcdef");
	std::string r;

	for (size_t i = 0; i < len; ++i) {
		const unsigned char h = s[i] / 16;
		const unsigned char l = s[i] % 16;

		r = r + b.substr(h, 1) + b.substr(l, 1);
	}

	return r;
}

static void printResult(const uint64_t seed[4], uint64_t round, result r, uint8_t score) {
	// Format private key
	uint64_t carry = 0;
	uint64_t seedRes[4];

	seedRes[0] = seed[0] + round; 
	carry = seedRes[0] < round;
	seedRes[1] = seed[1] + carry; 
	carry = !seedRes[1];
	seedRes[2] = seed[2] + carry; 
	carry = !seedRes[2];
	seedRes[3] = seed[3] + carry + r.foundId;

	std::ostringstream ss;
	ss << std::hex << std::setfill('0');
	ss << std::setw(16) << seedRes[3] << std::setw(16) << seedRes[2] << std::setw(16) << seedRes[1] << std::setw(16) << seedRes[0];
	const std::string strPrivate = ss.str();

	// Format public key
	const std::string strPublic = toHex(r.foundHash, 20);

	// Print
	std::cout << "s Score: " << std::setw(2) << (int)score << " Private: 0x" << strPrivate << ' ';

	std::cout << ": 0x" << strPublic << std::endl;
}

int main(int argc, char ** argv)
{

		ArgParser argp(argc, argv);
		bool bHelp = false;
		bool bModeBenchmark = false;
		bool bModeZeros = false;
		bool bModeZeroBytes = false;
		bool bModeLetters = false;
		bool bModeNumbers = false;
		std::string strModeLeading;
		std::string strModeMatching;
		std::string strPublicKey;
		bool bModeLeadingRange = false;
		bool bModeRange = false;
		bool bModeMirror = false;
		bool bModeDoubles = false;
		int rangeMin = 0;
		int rangeMax = 0;
		std::vector<size_t> vDeviceSkipIndex;
		size_t worksizeLocal = 64;
		size_t worksizeMax = 0; // Will be automatically determined later if not overriden by user
		bool bNoCache = false;
		size_t inverseSize = 255;
		size_t inverseMultiple = 16384;
		bool bMineContract = false;

		argp.addSwitch('h', "help", bHelp);
		argp.addSwitch('0', "benchmark", bModeBenchmark);
		argp.addSwitch('1', "zeros", bModeZeros);
		argp.addSwitch('2', "letters", bModeLetters);
		argp.addSwitch('3', "numbers", bModeNumbers);
		argp.addSwitch('4', "leading", strModeLeading);
		argp.addSwitch('5', "matching", strModeMatching);
		argp.addSwitch('6', "leading-range", bModeLeadingRange);
		argp.addSwitch('7', "range", bModeRange);
		argp.addSwitch('8', "mirror", bModeMirror);
		argp.addSwitch('9', "leading-doubles", bModeDoubles);
		argp.addSwitch('m', "min", rangeMin);
		argp.addSwitch('M', "max", rangeMax);
		argp.addMultiSwitch('s', "skip", vDeviceSkipIndex);
		argp.addSwitch('w', "work", worksizeLocal);
		argp.addSwitch('W', "work-max", worksizeMax);
		argp.addSwitch('n', "no-cache", bNoCache);
		argp.addSwitch('i', "inverse-size", inverseSize);
		argp.addSwitch('I', "inverse-multiple", inverseMultiple);
		argp.addSwitch('c', "contract", bMineContract);
		argp.addSwitch('z', "publicKey", strPublicKey);
		argp.addSwitch('b', "zero-bytes", bModeZeroBytes);

		if (!argp.parse()) {
			std::cout << "error: bad arguments, -h for help" << std::endl;
			return 1;
		}
        if (bHelp) {
            std::cout << g_strHelp << std::endl;
            return 0;
        }
		if (strPublicKey.length() == 0) {
			std::cout << "error: this tool requires your public key to derive it's private key security" << std::endl;
			return 1;
		}
	cudaError_t error;
	int n = 256;
	if(argc > 1)	{ n = atoi(argv[1]);}     // Number of particles
	if(argc > 2)	{	srand(atoi(argv[2])); } // Random seed

	error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
  	    printf("0 %s\n",cudaGetErrorString(error));
  	    exit(1);
  	}

	particle * pArray = new particle[n];
	particle* devPArray = NULL;
	point * precomp = NULL;
	mp_number* pointsDeltaX = NULL;
	mp_number* prevLambda = NULL;
	mp_number* invData = NULL;
	cudaMalloc(&devPArray, n*sizeof(particle));
	cudaMalloc(&precomp, 8160 * sizeof(point));
	cudaMalloc(&pointsDeltaX, PROFANITY_INVERSE_SIZE * n * sizeof(mp_number));
	cudaMalloc(&prevLambda, PROFANITY_INVERSE_SIZE * n * sizeof(mp_number));
	cudaMalloc(&invData, PROFANITY_INVERSE_SIZE * n * sizeof(mp_number));

	cudaDeviceSynchronize(); error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
        printf("1 %s\n",cudaGetErrorString(error));
        exit(1);
  	}

	mp_number* pointsDeltaXHost = new mp_number[PROFANITY_INVERSE_SIZE * n];
	for(int i=0; i< PROFANITY_INVERSE_SIZE * n; i++)
	{
		for(int j=0; j<8; j++)
		{
			pointsDeltaXHost[i].d[j] = 0;
		}
	}

	mp_number* prevLambdaHost = new mp_number[PROFANITY_INVERSE_SIZE * n];
	for(int i=0; i< PROFANITY_INVERSE_SIZE * n; i++)
	{
		for(int j=0; j<8; j++)
		{
			prevLambdaHost[i].d[j] = 0;
		}
	}

	mp_number* invDataHost = new mp_number[PROFANITY_INVERSE_SIZE * n];
	for(int i=0; i< PROFANITY_INVERSE_SIZE * n; i++)
	{
		for(int j=0; j<8; j++)
		{
			invDataHost[i].d[j] = 0;
		}
	}




	cudaMemcpy(devPArray, pArray, n*sizeof(particle), cudaMemcpyHostToDevice);
	cudaMemcpy(precomp, g_precomp, 8160 * sizeof(point), cudaMemcpyHostToDevice);
	cudaMemcpy(pointsDeltaX, pointsDeltaXHost, PROFANITY_INVERSE_SIZE * n*sizeof(mp_number), cudaMemcpyHostToDevice);
	cudaMemcpy(prevLambda, prevLambdaHost, PROFANITY_INVERSE_SIZE * n*sizeof(mp_number), cudaMemcpyHostToDevice);
	cudaMemcpy(invData, invDataHost, PROFANITY_INVERSE_SIZE * n*sizeof(mp_number), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize(); error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
        printf("2 %s\n",cudaGetErrorString(error));
        exit(1);
  	}

	float dt = (float)rand()/(float) RAND_MAX; // Random distance each step
	advanceParticles<<< 1, 256>>>(dt, devPArray, n, precomp, pointsDeltaX, prevLambda, invData);
	error = cudaGetLastError();
	if (error != cudaSuccess)
    {
    printf("3 %s\n",cudaGetErrorString(error));
    exit(1);
    }

	cudaDeviceSynchronize();

	printf("Size of mp_number %d\n", sizeof(mp_number));
	cudaMemcpy(pArray, devPArray, n*sizeof(particle), cudaMemcpyDeviceToHost);
	cudaMemcpy(pointsDeltaXHost, pointsDeltaX, PROFANITY_INVERSE_SIZE * n*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(prevLambdaHost, prevLambda, PROFANITY_INVERSE_SIZE * n*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(invDataHost, invData, PROFANITY_INVERSE_SIZE * n*sizeof(mp_number), cudaMemcpyDeviceToHost);




	/*for (int n = 0; n < 100; n++)
	{
		printf("Hash no: %d\n", n);
		for (int i = 0; i < 32; i++)
		{
			printf("%d ", pArray[n].m_data[i]);
		}
		printf("\n");
	}*/
	for (int n = 0; n < 100; n++)
	{
		printf("Hash no: %d\n0x", n);
		const uint8_t* hash = (uint8_t * )invDataHost[n * PROFANITY_INVERSE_SIZE].d;
		const uint64_t seed[4] = {0, 0, 0, n};
		result r = {0};
		r.found = 1;
		r.foundId = n;
		memcpy(r.foundHash, hash, 20);
		printResult(seed, 0, r, 0);
		for (int i = 0; i < 20; i++)
		{
			printf("%02x", hash[i]);
		}
		printf("\n");
	}





	return 0;
}
