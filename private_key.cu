#include "create3.h"
#include "utils.hpp"
#include "Logger.hpp"
#include <iostream>
#include <cuda_runtime_api.h>
#include <string>
#include <filesystem>
#include <cstring>


__global__ void private_search(search_result* const results, int rounds)
{
    const size_t id = (threadIdx.x + blockIdx.x * blockDim.x);


    int round_no = 3;
    for (int i = 0; i < 20; i++) {
        results[id].addr[i] = 0x01;
    }
    results[id].id = id;
    results[id].round = round_no;


}

void run_kernel_private_search(private_search_data * data) {
    private_search<<<(int)(data->kernel_groups), data->kernel_group_size>>>(data->device_result, data->rounds);
}

