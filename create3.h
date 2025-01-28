#pragma once

#include "keccak.h"

typedef struct {
    uint8_t b[20];
} factory;

typedef union {
    uint8_t b[32];
    uint32_t d[8];
    uint64_t q[4];
} salt;



void test_create3();
