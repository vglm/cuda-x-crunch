#pragma once

#include "create3.h"
#include <string>
// private key search

#define USE_PREV_LAMBDA_GLOBAL
#define PROFANITY_INVERSE_SIZE 255
#define RESULTS_ARRAY_SIZE 64 * 1024

struct private_search_data {
    cl_ulong4 public_key_x;
    cl_ulong4 public_key_y;
    cl_ulong4 seed;
    int rounds;
    int kernel_group_size;
    int kernel_groups;
    search_result * device_result;
    search_result * host_result;
#ifdef USE_PREV_LAMBDA_GLOBAL
    mp_number * device_pInverse;
    mp_number * device_deltaX;
    mp_number * device_prev_lambda;
#endif
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


