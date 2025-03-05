#pragma once

#include "types.hpp"
#include <string>

#include <cstdint>

struct factory {
    uint8_t b[20];
};

struct search_result {
    uint8_t addr[20];
    uint32_t id;
    uint32_t round;
};

struct create3_search_data {
    char factory[41];
    char outputDir[1024];
    int rounds;
    int kernel_group_size;
    int kernel_groups;
    search_result * device_result;
    search_result * host_result;
    uint64_t total_compute;
    double time_started;
};

union salt {
    uint8_t b[32];
    uint32_t d[8];
    uint64_t q[4];
};

union ethhash {
    uint8_t b[200];
    uint32_t d[50];
    uint64_t q[25];
};

void load_seed_to_device(salt *seed_data);
void load_factory_to_device(const char* factory);
void create3_data_init(create3_search_data* data);
void create3_data_destroy(create3_search_data* data);
void create3_search(create3_search_data* factory);
#ifdef UNUSED_OLD_TESTS
void test_create3();
#endif

//cuda types
void update_device_factory(const uint8_t* factory);
void update_device_salt(const salt* salt);
void run_kernel_create3_search(create3_search_data * data);

// private key search

#define PROFANITY_INVERSE_SIZE 10

struct private_search_data {
    cl_ulong4 public_key_x;
    cl_ulong4 public_key_y;
    cl_ulong4 seed;
    int rounds;
    int kernel_group_size;
    int kernel_groups;
    search_result * device_result;
    search_result * host_result;
    mp_number * device_pInverse;
    mp_number * device_deltaX;
    mp_number * device_prev_lambda;
    point * device_precomp;
    uint64_t total_compute;
    double time_started;
};


void update_public_key(mp_number const& x, mp_number const& y);
void private_data_init(private_search_data *init_data);
void private_data_search(std::string public_key, private_search_data *init_data);
void private_data_destroy(private_search_data *init_data);
void run_kernel_private_search(private_search_data * data);

salt generate_random_salt();


