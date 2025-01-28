#pragma once

#include "keccak.h"

typedef struct {
    uint8_t b[20];
} factory;

typedef struct {
    uint8_t b[32];
} salt;



void test_create3();
