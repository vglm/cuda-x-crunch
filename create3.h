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

typedef struct {
    char factory[41];
    int rounds;
    int kernel_group_size;
    int kernel_groups;
    search_result * device_result;
    search_result * host_result;
    uint64_t total_compute;
    uint64_t time_started;
} create3_search_data;


typedef union {
    uint8_t b[32];
    uint32_t d[8];
    uint64_t q[4];
} salt;

void create3_data_init(const char* factory, create3_search_data* data);
void create3_data_destroy(create3_search_data* data);
void create3_search(create3_search_data* factory);
void test_create3();
