#pragma once

#define MP_WORDS 8
#define MP_BITS 32

typedef unsigned int mp_word;
typedef struct {
	mp_word d[MP_WORDS];
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
} cl_ulong4;
