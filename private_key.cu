#include "create3.h"

__constant__ mp_number g_publicKeyX = {0};
__constant__ mp_number g_publicKeyY = {0};

void update_public_key(const mp_number &x, const mp_number &y)
{
    cudaMemcpyToSymbol(g_publicKeyX, &x, sizeof(mp_number));
    cudaMemcpyToSymbol(g_publicKeyY, &y, sizeof(mp_number));
}

__global__ void private_search(search_result* const results, int rounds)
{
    const size_t id = (threadIdx.x + blockIdx.x * blockDim.x);


    int round_no = 3;
    for (int i = 0; i < 20; i++) {
        results[id].addr[i] = g_publicKeyX.b[i];
    }
    if (id == 100) {
        results[id].id = id;
        results[id].round = round_no;
    }


}

void run_kernel_private_search(private_search_data * data) {
    private_search<<<(int)(data->kernel_groups), data->kernel_group_size>>>(data->device_result, data->rounds);
}

