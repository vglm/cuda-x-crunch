#include "create3.h"

#define ROL(X, S) (((X) << S) | ((X) >> (64 - S)))

#define THETA_(M, N, O) t = b[M] ^ ROL(b[N], 1); \
a[O + 0] = a[O + 0] ^ t; a[O + 5] = a[O + 5] ^ t; a[O + 10] = a[O + 10] ^ t; \
a[O + 15] = a[O + 15] ^ t; a[O + 20] = a[O + 20] ^ t;

#define THETA() \
b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]; \
b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]; \
b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]; \
b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]; \
b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]; \
THETA_(4, 1, 0); THETA_(0, 2, 1); THETA_(1, 3, 2); THETA_(2, 4, 3); THETA_(3, 0, 4);

#define RHO_PI_(M, N) t = b[0]; b[0] = a[M]; a[M] = ROL(t, N);

#define RHO_PI() t = a[1]; b[0] = a[10]; a[10] = ROL(t, 1); \
RHO_PI_(7, 3); RHO_PI_(11, 6); RHO_PI_(17, 10); RHO_PI_(18, 15); RHO_PI_(3, 21); RHO_PI_(5, 28); \
RHO_PI_(16, 36); RHO_PI_(8, 45); RHO_PI_(21, 55); RHO_PI_(24, 2); RHO_PI_(4, 14); RHO_PI_(15, 27); \
RHO_PI_(23, 41); RHO_PI_(19, 56); RHO_PI_(13, 8); RHO_PI_(12, 25); RHO_PI_(2, 43); RHO_PI_(20, 62); \
RHO_PI_(14, 18); RHO_PI_(22, 39); RHO_PI_(9, 61); RHO_PI_(6, 20); RHO_PI_(1, 44);

#define CHI_(N) \
b[0] = a[N + 0]; b[1] = a[N + 1]; b[2] = a[N + 2]; b[3] = a[N + 3]; b[4] = a[N + 4]; \
a[N + 0] = b[0] ^ ((~b[1]) & b[2]); \
a[N + 1] = b[1] ^ ((~b[2]) & b[3]); \
a[N + 2] = b[2] ^ ((~b[3]) & b[4]); \
a[N + 3] = b[3] ^ ((~b[4]) & b[0]); \
a[N + 4] = b[4] ^ ((~b[0]) & b[1]);

#define CHI() CHI_(0); CHI_(5); CHI_(10); CHI_(15); CHI_(20);

#define IOTA(X) a[0] = a[0] ^ X;

#define ITER(X) THETA(); RHO_PI(); CHI(); IOTA(X);

#define ITERS() \
ITER(0x0000000000000001); ITER(0x0000000000008082); \
ITER(0x800000000000808a); ITER(0x8000000080008000); \
ITER(0x000000000000808b); ITER(0x0000000080000001); \
ITER(0x8000000080008081); ITER(0x8000000000008009); \
ITER(0x000000000000008a); ITER(0x0000000000000088); \
ITER(0x0000000080008009); ITER(0x000000008000000a); \
ITER(0x000000008000808b); ITER(0x800000000000008b); \
ITER(0x8000000000008089); ITER(0x8000000000008003); \
ITER(0x8000000000008002); ITER(0x8000000000000080); \
ITER(0x000000000000800a); ITER(0x800000008000000a); \
ITER(0x8000000080008081); ITER(0x8000000000008080); \
ITER(0x0000000080000001); ITER(0x8000000080008008);

void cpu_compute_keccak_full(ethhash* hash) {
    uint64_t b[5];
    uint64_t t;
    uint64_t* a = (uint64_t*)hash;
    ITERS();
}
void cpu_partial_keccakf(uint64_t* a)
{
    uint64_t b[5];
    uint64_t t;
    ITER(0x0000000000000001); ITER(0x0000000000008082);
    ITER(0x800000000000808a); ITER(0x8000000080008000);
    ITER(0x000000000000808b); ITER(0x0000000080000001);
    ITER(0x8000000080008081); ITER(0x8000000000008009);
    ITER(0x000000000000008a); ITER(0x0000000000000088);
    ITER(0x0000000080008009); ITER(0x000000008000000a);
    ITER(0x000000008000808b); ITER(0x800000000000008b);
    ITER(0x8000000000008089); ITER(0x8000000000008003);
    ITER(0x8000000000008002); ITER(0x8000000000000080);
    ITER(0x000000000000800a); ITER(0x800000008000000a);
    ITER(0x8000000080008081); ITER(0x8000000000008080);
    ITER(0x0000000080000001);

    // iteration 24 (partial)
#define o ((uint32_t *)(a))
  // Theta (partial)
    b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

    a[0] ^= b[4] ^ ROL(b[1], 1u);
    a[6] ^= b[0] ^ ROL(b[2], 1u);
    a[12] ^= b[1] ^ ROL(b[3], 1u);
    a[18] ^= b[2] ^ ROL(b[4], 1u);
    a[24] ^= b[3] ^ ROL(b[0], 1u);

    // Rho Pi (partial)
    o[3] = (o[13] >> 20) | (o[12] << 12);
    a[2] = ROL(a[12], 43);
    a[3] = ROL(a[18], 21);
    a[4] = ROL(a[24], 14);

    // Chi (partial)
    o[3] ^= ((~o[5]) & o[7]);
    o[4] ^= ((~o[6]) & o[8]);
    o[5] ^= ((~o[7]) & o[9]);
    o[6] ^= ((~o[8]) & o[0]);
    o[7] ^= ((~o[9]) & o[1]);
#undef o
}

uint8_t g_factory[20] = { 0 };
salt g_randomSalt = { 0 };

void cpu_update_device_factory(const uint8_t* factory)
{
    //cudaMemcpyToSymbol(g_factory, factory, 20);
}

void cpu_update_device_salt(const salt* seed_data)
{
    //cudaMemcpyToSymbol(g_randomSalt, seed_data, sizeof(salt));
}


uint64_t g_search_prefix_contract = 0;

void cpu_update_search_prefix_contract(const uint64_t& pref)
{
    //cudaMemcpyToSymbol(g_search_prefix_contract, &pref, sizeof(uint64_t));
}


void cpu_create3_search_kernel(search_result* const results, int rounds, const size_t job_idx)
{
    const size_t id = job_idx;

    //test

    for (int round = 1; round <= rounds; round++) {
        ethhash first;

        first.b[0] = 0xff;
        for (int i = 0; i < 20; i++) {
            first.b[i + 1] = g_factory[i];
        }
        salt salt_data;
        salt_data.d[0] = id;
        salt_data.d[1] = round;
        salt_data.d[2] = g_randomSalt.d[2];
        salt_data.d[3] = g_randomSalt.d[3];
        salt_data.d[4] = g_randomSalt.d[4];
        salt_data.d[5] = g_randomSalt.d[5];
        salt_data.d[6] = g_randomSalt.d[6];
        salt_data.d[7] = g_randomSalt.d[7];

        for (int i = 0; i < 32; i++) {
            first.b[i + 21] = salt_data.b[i];
        }

        //0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f

        first.b[53] = 0x21;
        first.b[54] = 0xc3;
        first.b[55] = 0x5d;
        first.b[56] = 0xbe;
        first.b[57] = 0x1b;
        first.b[58] = 0x34;
        first.b[59] = 0x4a;
        first.b[60] = 0x24;
        first.b[61] = 0x88;
        first.b[62] = 0xcf;
        first.b[63] = 0x33;
        first.b[64] = 0x21;
        first.b[65] = 0xd6;
        first.b[66] = 0xce;
        first.b[67] = 0x54;
        first.b[68] = 0x2f;
        first.b[69] = 0x8e;
        first.b[70] = 0x9f;
        first.b[71] = 0x30;
        first.b[72] = 0x55;
        first.b[73] = 0x44;
        first.b[74] = 0xff;
        first.b[75] = 0x09;
        first.b[76] = 0xe4;
        first.b[77] = 0x99;
        first.b[78] = 0x3a;
        first.b[79] = 0x62;
        first.b[80] = 0x31;
        first.b[81] = 0x9a;
        first.b[82] = 0x49;
        first.b[83] = 0x7c;
        first.b[84] = 0x1f;
        first.b[85] = 0x01;
        //total length 85
        for (int i = 86; i < 135; ++i)
            first.b[i] = 0;

        first.b[135] = 0x80;
        for (int i = 136; i < 200; ++i)
            first.b[i] = 0;
        cpu_compute_keccak_full(&first);

        first.b[0] = 0xd6;
        first.b[1] = 0x94;
        for (int i = 12; i < 32; i++) {
            first.b[i - 10] = first.b[i];
        }

        first.b[22] = 0x01;
        first.b[23] = 0x01;
        for (int i = 24; i < 135; ++i)
            first.b[i] = 0;
        first.b[135] = 0x80;
        for (int i = 136; i < 200; ++i)
            first.b[i] = 0;

        cpu_partial_keccakf((uint64_t*)&first);

        ethaddress& addr = *(ethaddress*)&first.b[12];
        /*
        if (scorer(addr, g_search_prefix_contract) == SCORE_ACCEPTED) {
            results[id].round = round;
            results[id].id = id;

            for (int i = 0; i < 20; i++) {
                results[id].addr[i] = first.b[i + 12];
            }
        }*/


    }
}

void run_cpu_create3_search(create3_search_data * data) {
    //create3_search_kernel<<<(int)(data->kernel_groups), data->kernel_group_size>>>(data->device_result, data->rounds);
}
