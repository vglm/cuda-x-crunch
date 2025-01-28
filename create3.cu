#include "create3.h"

__global__ void create3_host(factory* const factory_data, salt* const salt_data)
{
	const size_t id = (threadIdx.x + blockIdx.x * blockDim.x);


}


void test_create3()
{
    const int kernel_group_size = 64;
    const int data_count = 100000 * kernel_group_size;
    printf("Generating test data %d...\n", data_count);


    salt* s = new salt[data_count]();
    {
        uint8_t salt[32];
        const char* test_salt = "6233362f90c89a3be581c4000000000000000000000000000000000000000000";
        for (int i = 0; i < 32; i++) {
            unsigned int byte;
            sscanf(test_salt + i * 2, "%2x", &byte); // Parse 2-character hex string
            salt[i] = (uint8_t)(byte);
        }
        for (int n = 0; n < data_count; n++) {
            memcpy(s[n].b, salt, 32);
        }
    }

    factory* f = new factory[data_count]();
    {
        uint8_t factory[20];
        const char* test_factory = "9e3f8eae49e442a323ef2094f277bf62752e6995";
        for (int i = 0; i < 20; i++) {
            unsigned int byte;
            sscanf(test_factory + i * 2, "%2x", &byte); // Parse 2-character hex string
            factory[i] = (uint8_t)(byte);
        }
        for (int n = 0; n < data_count; n++) {
            memcpy(f[n].b, factory, 20);
        }
    }



    salt* deviceSalt = NULL;
    cudaMalloc(&deviceSalt, sizeof(salt) * data_count);
    factory* deviceFactory = NULL;
    cudaMalloc(&deviceFactory, sizeof(factory) * data_count);

    printf("Copying data to device %d MB...\n", (uint32_t)(sizeof(salt) * data_count / 1024 / 1024 + sizeof(factory) * data_count / 1024 / 1024));

    cudaMemcpy(deviceSalt, s, sizeof(salt) * data_count, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFactory, f, sizeof(factory) * data_count, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Initialize keccak test failed %s\n",cudaGetErrorString(error));
        exit(1);
    }
    printf("Running keccak kernel...\n");
    auto start = std::chrono::high_resolution_clock::now();
    const uint64_t current_time = time(NULL);
    create3_host<<<data_count / kernel_group_size, kernel_group_size>>>(deviceFactory, deviceSalt);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Output the duration
    std::cout << "Time taken: " << duration.count() / 1000.0 / 1000.0 << " ms" << std::endl;

    printf("Start data factory: ");
    for (int i = 0; i < 20; i++) {
        printf("%02x", f[0].b[i]);
    }
    printf("\n");
    printf("Start data salt: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", s[0].b[i]);
    }
    printf("\n");


    printf("Copying data back...\n");
    cudaMemcpy(f, deviceFactory, data_count * sizeof(factory), cudaMemcpyDeviceToHost);
    cudaMemcpy(s, deviceSalt, data_count * sizeof(salt), cudaMemcpyDeviceToHost);

    printf("Freeing device memory...\n");
    cudaFree(deviceFactory);
    cudaFree(deviceSalt);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Initialize keccak test failed %s\n",cudaGetErrorString(error));
        exit(1);
    }

    printf("Checking results...\n");
    for (int n = 0; n < data_count; n++) {
        char hexAddr[43] = { 0 };
        hexAddr[0] = '0';
        hexAddr[1] = 'x';
        for (int i = 0; i < 20; i++) {
            sprintf(&hexAddr[(i) * 2 + 2], "%02x", f[n].b[i]);
        }
        if (strcmp(hexAddr, "0x35743574e8474571a6bf34621f185a917e11d919") != 0) {
            printf("Keccak test failed expected %s got %s\n", "0x35743574e8474571a6bf34621f185a917e11d919", hexAddr);
            exit(1);
        }
    }
    printf("Keccak test passed\n");

    printf("Freeing host memory...\n");
    delete[] f;
    delete[] s;
}