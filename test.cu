/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
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
#include "particle.h"
#include <stdlib.h>
#include <stdio.h>

typedef union {
	uint8_t b[200];
	uint64_t q[25];
	uint32_t d[50];
} ethhash;

#define rotate(x, y) __funnelshift_r(x, x, y)

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


__global__ void advanceParticles(float dt, particle * pArray, int nParticles)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < nParticles)
	{
		for (int i = 0; i < 32; i++)
		{
			pArray[idx].m_data[i] = idx + i;
		}

	}
}

int main(int argc, char ** argv)
{
	cudaError_t error;
	int n = 1000000;
	if(argc > 1)	{ n = atoi(argv[1]);}     // Number of particles
	if(argc > 2)	{	srand(atoi(argv[2])); } // Random seed

	error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
  	    printf("0 %s\n",cudaGetErrorString(error));
  	    exit(1);
  	}

	particle * pArray = new particle[n];
	particle * devPArray = NULL;
	cudaMalloc(&devPArray, n*sizeof(particle));
	cudaDeviceSynchronize(); error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
        printf("1 %s\n",cudaGetErrorString(error));
        exit(1);
  	}

	cudaMemcpy(devPArray, pArray, n*sizeof(particle), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize(); error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
        printf("2 %s\n",cudaGetErrorString(error));
        exit(1);
  	}

	for(int i=0; i<100; i++)
	{
		float dt = (float)rand()/(float) RAND_MAX; // Random distance each step
		advanceParticles<<< 1 +  n/256, 256>>>(dt, devPArray, n);
		error = cudaGetLastError();
		if (error != cudaSuccess)
    	{
    	printf("3 %s\n",cudaGetErrorString(error));
    	exit(1);
    	}

		cudaDeviceSynchronize();
	}
	cudaMemcpy(pArray, devPArray, n*sizeof(particle), cudaMemcpyDeviceToHost);

	for (int n = 0; n < 100; n++)
	{
		printf("Hash no: %d\n", n);
		for (int i = 0; i < 32; i++)
		{
			printf("%d ", pArray[n].m_data[i]);
		}
		printf("\n");
	}
	return 0;
}
