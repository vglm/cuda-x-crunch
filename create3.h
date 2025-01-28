#pragma once

#include "keccak.h"

typedef struct {
    uint8_t b[20];
} factory;

typedef struct {
    uint8_t addr[20];
    uint32_t id;
    uint32_t round;
} search_result;

typedef union {
    uint8_t b[32];
    uint32_t d[8];
    uint64_t q[4];
} salt;


void create3_search(const char* factory);
void test_create3();
