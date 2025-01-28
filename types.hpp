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
