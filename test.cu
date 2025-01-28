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
#include "keccak.h"
#include "create3.h"
#include "help.hpp"
#include "utils.hpp"
#include "ArgParser.hpp"
#include "debug_utils.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "precomp.hpp"
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>


__device__ const mp_number tripleNegativeGx = { {0xbb17b196, 0xf2287bec, 0x76958573, 0xf82c096e, 0x946adeea, 0xff1ed83e, 0x1269ccfa, 0x92c4cc83 } };
__device__ const mp_number negativeGy       = { {0x04ef2777, 0x63b82f6f, 0x597aabe6, 0x02e84bb7, 0xf1eef757, 0xa25b0403, 0xd95c3b9a, 0xb7c52588 } };


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


__device__ void profanity_init(const point* const precomp, mp_number* const pDeltaX, mp_number* const pPrevLambda, result* const pResult, const uint64_t seed[4], const uint64_t seedX[4], const uint64_t seedY[4]) {
	const size_t id = (threadIdx.x + blockIdx.x * blockDim.x);

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
	p.x.d[0] = seedX[0] & 0xFFFFFFFF;
	p.x.d[1] = seedX[0] >> 32;
	p.x.d[2] = seedX[1] & 0xFFFFFFFF;
	p.x.d[3] = seedX[1] >> 32;
	p.x.d[4] = seedX[2] & 0xFFFFFFFF;
	p.x.d[5] = seedX[2] >> 32;
	p.x.d[6] = seedX[3] & 0xFFFFFFFF;
	p.x.d[7] = seedX[3] >> 32;
	p.y.d[0] = seedY[0] & 0xFFFFFFFF;
	p.y.d[1] = seedY[0] >> 32;
	p.y.d[2] = seedY[1] & 0xFFFFFFFF;
	p.y.d[3] = seedY[1] >> 32;
	p.y.d[4] = seedY[2] & 0xFFFFFFFF;
	p.y.d[5] = seedY[2] >> 32;
	p.y.d[6] = seedY[3] & 0xFFFFFFFF;
	p.y.d[7] = seedY[3] >> 32;

	point p_random;
	bool bIsFirst = true;

	mp_number tmp1, tmp2;
	point tmp3;

	// Calculate k*G where k = seed.wzyx (in other words, find the point indicated by the private key represented in seed)
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 0, seed[0]);
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 1, seed[1]);
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 2, seed[2]);
	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 3, seed[3] + id);

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
__global__ void profanity_inverse(const mp_number* const pDeltaX, mp_number* const pInverse) {
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

__global__ void profanity_iterate(mp_number* const pDeltaX, mp_number* const pInverse, mp_number* const pPrevLambda) {
	const size_t id = (threadIdx.x + blockIdx.x * blockDim.x);

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

__global__ void advanceParticlesPart1(float dt, point* precomp, mp_number* pointsDeltaX, mp_number* pPrevLambda, mp_number* pInverse,
    uint64_t seedX1, uint64_t seedX2, uint64_t seedX3, uint64_t seedX4, uint64_t seedY1, uint64_t seedY2, uint64_t seedY3, uint64_t seedY4)
{
	uint64_t seed[4];
	seed[0] = 3;
	seed[1] = 1;
	seed[2] = 1;
	seed[3] = 1;
	uint64_t seedX[4];
	seedX[0] = seedX1;
	seedX[1] = seedX2;
	seedX[2] = seedX3;
	seedX[3] = seedX4;
	uint64_t seedY[4];
	seedY[0] = seedY1;
	seedY[1] = seedY2;
	seedY[2] = seedY3;
	seedY[3] = seedY4;
	result pResult = { 0 };



	profanity_init(precomp, pointsDeltaX, pPrevLambda, &pResult, seed, seedX, seedY);


	//pInverse[(threadIdx.x + blockIdx.x * blockDim.x) * PROFANITY_INVERSE_SIZE].d[0] = (uint32_t)(seedX1 & 0x00000000FFFFFFFFU);
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

struct PublicKeyPart {
    uint64_t val[4];
};

PublicKeyPart hexStringToUint64(const std::string& hexStr) {
    // Ensure the input string is not too long
    if (hexStr.length() != 64) {
        throw std::invalid_argument("Hex string has to be exactly 64 characters.");
    }
    // Function to convert a 16-character substring to a uint64_t
    auto hexToUint64 = [](const std::string& subHex) -> uint64_t {
        uint64_t value = 0;
        std::istringstream iss(subHex);
        iss >> std::hex >> value;
        return value;
    };

    PublicKeyPart result;
    // Extract 16-character chunks and convert them to uint64_t
    result.val[0] = hexToUint64(hexStr.substr(0, 16));
    result.val[1] = hexToUint64(hexStr.substr(16, 16));
    result.val[2] = hexToUint64(hexStr.substr(32, 16));
    result.val[3] = hexToUint64(hexStr.substr(48, 16));
    return result;
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
    if (bModeBenchmark) {
        test_create3();
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
    strPublicKey = string_replace(strPublicKey, "0x", "");
    if (strPublicKey.length() != 128) {
        std::cout << "error: public key must be 128 hexadecimal characters long" << std::endl;
        return 1;
    }


    PublicKeyPart publicKeyX = hexStringToUint64(strPublicKey.substr(0, 64));
    PublicKeyPart publicKeyY = hexStringToUint64(strPublicKey.substr(64, 64));



	cudaError_t error;
	const int run_size = 256;

	error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
  	    printf("0 %s\n",cudaGetErrorString(error));
  	    exit(1);
  	}

	point * precomp = NULL;
	mp_number* pointsDeltaX = NULL;
	mp_number* prevLambda = NULL;
	mp_number* invData = NULL;


	cudaMalloc(&precomp, 8160 * sizeof(point));
	cudaMalloc(&pointsDeltaX, PROFANITY_INVERSE_SIZE * run_size * sizeof(mp_number));
	cudaMalloc(&prevLambda, PROFANITY_INVERSE_SIZE * run_size * sizeof(mp_number));
	cudaMalloc(&invData, PROFANITY_INVERSE_SIZE * run_size * sizeof(mp_number));



	cudaDeviceSynchronize(); error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
        printf("1 %s\n",cudaGetErrorString(error));
        exit(1);
  	}

	mp_number* pointsDeltaXHost = new mp_number[PROFANITY_INVERSE_SIZE * run_size];
	for(int i=0; i< PROFANITY_INVERSE_SIZE * run_size; i++)
	{
		for(int j=0; j<8; j++)
		{
			pointsDeltaXHost[i].d[j] = 0;
		}
	}

	mp_number* prevLambdaHost = new mp_number[PROFANITY_INVERSE_SIZE * run_size];
	for(int i=0; i< PROFANITY_INVERSE_SIZE * run_size; i++)
	{
		for(int j=0; j<8; j++)
		{
			prevLambdaHost[i].d[j] = 0;
		}
	}

	mp_number* invDataHost = new mp_number[PROFANITY_INVERSE_SIZE * run_size];
	for(int i=0; i< PROFANITY_INVERSE_SIZE * run_size; i++)
	{
		for(int j=0; j<8; j++)
		{
			invDataHost[i].d[j] = 0;
		}
	}


	cudaMemcpy(precomp, g_precomp, 8160 * sizeof(point), cudaMemcpyHostToDevice);
	cudaMemcpy(pointsDeltaX, pointsDeltaXHost, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyHostToDevice);
	cudaMemcpy(prevLambda, prevLambdaHost, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyHostToDevice);
	cudaMemcpy(invData, invDataHost, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize(); error = cudaGetLastError();
	if (error != cudaSuccess)
  	{
        printf("2 %s\n",cudaGetErrorString(error));
        exit(1);
  	}

	float dt = (float)rand()/(float) RAND_MAX; // Random distance each step
	advanceParticlesPart1<<< 1, 256>>>(dt, precomp, pointsDeltaX, prevLambda, invData,
	publicKeyX.val[0],
	publicKeyX.val[1],
    	publicKeyX.val[2],
        	publicKeyX.val[3],
            	publicKeyY.val[1],
            	publicKeyY.val[2],
            	publicKeyY.val[3],
            	publicKeyY.val[4]
	);
	cudaDeviceSynchronize();


	cudaMemcpy(pointsDeltaXHost, pointsDeltaX, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(prevLambdaHost, prevLambda, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(invDataHost, invData, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("Delta host no: %d\n", i);
        pretty_print_mp_number(pointsDeltaXHost[i]);
        printf("\n");
    }
    for (int i = 0; i < 10; i++) {
        printf("Prev prevLambdaHost: %d\n", i);
        pretty_print_mp_number(prevLambdaHost[i]);
        printf("\n");
    }
    for (int i = 0; i < 10; i++) {
        printf("Inv data no: %d\n", i);
        pretty_print_mp_number(invDataHost[i]);
        printf("\n");
    }

	cudaDeviceSynchronize();
	profanity_inverse<<< 1, 1>>>(pointsDeltaX, invData);



    error = cudaGetLastError();

	cudaMemcpy(pointsDeltaXHost, pointsDeltaX, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(prevLambdaHost, prevLambda, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(invDataHost, invData, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("Delta host no: %d\n", i);
        pretty_print_mp_number(pointsDeltaXHost[i]);
        printf("\n");
    }
    for (int i = 0; i < 10; i++) {
        printf("Prev prevLambdaHost: %d\n", i);
        pretty_print_mp_number(prevLambdaHost[i]);
        printf("\n");
    }
    for (int i = 0; i < 10; i++) {
        printf("Inv data no: %d\n", i);
        pretty_print_mp_number(invDataHost[i]);
        printf("\n");
    }

    if (error != cudaSuccess)
    {
        printf("profanity_inverse error: %s\n",cudaGetErrorString(error));
        exit(1);
    }
    cudaDeviceSynchronize();

	profanity_iterate<<< 1, 256>>>(pointsDeltaX, invData, prevLambda);
	cudaDeviceSynchronize();
	profanity_inverse <<<  1, 1 >>>  (pointsDeltaX, invData);
	cudaDeviceSynchronize();
	profanity_iterate <<<  1, 256 >>>  (pointsDeltaX, invData, prevLambda);

	error = cudaGetLastError();
	if (error != cudaSuccess)
    {
        printf("profanity_iterate error %s\n",cudaGetErrorString(error));
        exit(1);
    }

    	cudaMemcpy(pointsDeltaXHost, pointsDeltaX, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
    	cudaMemcpy(prevLambdaHost, prevLambda, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
    	cudaMemcpy(invDataHost, invData, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 10; i++) {
            printf("Delta host no: %d\n", i);
            pretty_print_mp_number(pointsDeltaXHost[i]);
            printf("\n");
        }
        for (int i = 0; i < 10; i++) {
            printf("Prev prevLambdaHost: %d\n", i);
            pretty_print_mp_number(prevLambdaHost[i]);
            printf("\n");
        }
        for (int i = 0; i < 10; i++) {
            printf("Inv data no: %d\n", i);
            pretty_print_mp_number(invDataHost[i]);
            printf("\n");
        }

    //return 1;

	cudaDeviceSynchronize();

	printf("Size of mp_number %lld\n", sizeof(mp_number));

	cudaMemcpy(pointsDeltaXHost, pointsDeltaX, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(prevLambdaHost, prevLambda, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);
	cudaMemcpy(invDataHost, invData, PROFANITY_INVERSE_SIZE * run_size*sizeof(mp_number), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
    {
        printf("4 %s\n",cudaGetErrorString(error));
        exit(1);
    }



	/*for (int n = 0; n < 100; n++)
	{
		printf("Hash no: %d\n", n);
		for (int i = 0; i < 32; i++)
		{
			printf("%d ", pArray[n].m_data[i]);
		}
		printf("\n");
	}*/
	for (uint64_t n = 0; n < 10; n++)
	{
		printf("Hash no: %lld\n0x", n);
		const uint8_t* hash = (uint8_t * )invDataHost[n].d;
		const uint64_t seed[4] = {1, 1, 1, 1 + n};
		result r = {0};
		r.found = 1;
		r.foundId = (uint32_t) n;
		memcpy(r.foundHash, hash, 20);
		printResult(seed, 2, r, 0);
		for (int i = 0; i < 20; i++)
		{
			printf("%02x", hash[i]);
		}
		printf("\n");
	}





	return 0;
}
