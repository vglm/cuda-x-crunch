#pragma once

#include <cctype>

#define MP_WORDS 8
#define MP_BITS 32

typedef unsigned int mp_word;
typedef union {
    mp_word d[MP_WORDS];
    unsigned char b[32];
} mp_number;

/* ------------------------------------------------------------------------ */
/* Elliptic point and addition (with caveats).                              */
/* ------------------------------------------------------------------------ */
typedef struct {
	mp_number x;
	mp_number y;
} point;

typedef union
{
    struct {
        unsigned long long s[4];
    };
    struct {
    	mp_word d[MP_WORDS];
    };
    struct {
        unsigned long long  x, y, z, w;
    };
    struct {
        unsigned long long  s0, s1, s2, s3;
    };
    mp_number mpn;
} cl_ulong4;
