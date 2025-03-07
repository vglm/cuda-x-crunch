#pragma once
#include "create3.h"

#define SCORE_ACCEPTED 1
#define SCORE_REJECTED 0

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);  // Reverse byte order
}

#define MATCH3_PATTERN(a, b, c) \
    (let_full[i] == a && let_full[i + 1] == b && let_full[i + 2] == c)
#define MATCH4_PATTERN(a, b, c, d) \
    (let_full[i] == a && let_full[i + 1] == b && let_full[i + 2] == c && let_full[i + 3] == d)
#define MATCH5_PATTERN(a, b, c, d, e) \
    (let_full[i] == a && let_full[i + 1] == b && let_full[i + 2] == c && let_full[i + 3] == d && let_full[i + 4] == e)

__device__ inline uint32_t scorer(ethaddress& addr)
{
    uint8_t let_full[45];
    for (int i = 0; i < 20; i++) {
        let_full[2 * i] = (addr.b[i] >> 4) & 0x0f;
        let_full[2 * i + 1] = addr.b[i] & 0x0f;
    }
    let_full[40] = 0xff;
    let_full[41] = 0xff;
    let_full[42] = 0xff;
    let_full[43] = 0xff;
    let_full[44] = 0xff;


    int leading_score = 0;
    int group_score = 0;
    int letter_score = 0;
    int number_score = 0;
    int etherscan_score = 0;
    int pattern_score = 0;
    uint8_t first_letter = let_full[0];
    for (int i = 0; i < 40; i += 4) {
        if (*(uint32_t*)&let_full[0] == *(uint32_t*)&let_full[i]) {
            pattern_score += 1;
        }
        else {
            break;
        }
    }
    int patternbaba = 0;
    int dis = 0;
    int bb5numbers = 0;
    for (int i = 0; i < 40; i++) {
        if (MATCH5_PATTERN(0x0, 0x0, 0xb, 0xb, 0x5)) {
            bb5numbers += 1;
        }
        if (MATCH4_PATTERN(0x0, 0xb, 0xb, 0x5)) {
            bb5numbers += 2;
        }
        if (MATCH4_PATTERN(0xb, 0xb, 0x5, 0x0)) {
            bb5numbers += 2;
        }
        if (MATCH3_PATTERN(0xb, 0xb, 0x5)) {
            bb5numbers += 1;
        }
        if (MATCH3_PATTERN(0x0, 0x0, 0x0)) {
            bb5numbers += 1;
        }

    }


    for (int i = 0; i < 40; i++) {
        uint8_t letter = let_full[i];

        dis -= 1;
        if (i < 37) {
            if (dis < 0 && let_full[i + 0] == 0xb && let_full[i + 1] == 0xb && let_full[i + 2] == 0x5 && let_full[i + 3] == 0x0) {
                if (i == 0) {
                    patternbaba += 1;
                }
                if (i == 4) {
                    patternbaba += 1;
                }
                dis = 3;
            }
            if (dis < 0 && let_full[i + 0] == 0x0 && let_full[i + 1] == 0x0 && let_full[i + 2] == 0xb && let_full[i + 3] == 0x0b && let_full[i + 4] == 0x05 && let_full[i + 5] == 0x0 && let_full[i + 6] == 0x0) {


                    patternbaba += 2;

                dis = 3;
            }
            if (dis < 0 && let_full[i + 0] == 0x0 && let_full[i + 1] == 0xb && let_full[i + 2] == 0xb && let_full[i + 3] == 0x5) {
                if (i == 36) {
                    patternbaba += 1;
                }
                if (i == 32) {
                    patternbaba += 1;
                }
                dis = 3;
            }

       }
        if (leading_score < 50 && letter == first_letter) {
            leading_score += 1;
        }
        if (leading_score < 50 && letter != first_letter) {
            leading_score += 50;
        }
        if (i > 0 && letter == let_full[i - 1]) {
            group_score += 1;
        }
        if (letter >= 10) {
            letter_score += 1;
        }
        if (letter < 10) {
            number_score += 1;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (let_full[i] == let_full[i + 32]) {
            etherscan_score += 1;
        }
    }
    leading_score -= 50;

    int pattern = 0;
    int number = addr.d[0];



    if (number == bswap32(0xbadbabe0)
        || number == bswap32(0xb00bbabe)
        || number == bswap32(0xc0ffee00)
        || number == bswap32(0xdaedbeef)
        || number == bswap32(0x31415926)

        ) {
        pattern = 1;
    }


    if (
        pattern >= 1 ||
        bb5numbers >= 9 ||
        pattern_score >= 3 ||
        etherscan_score >= 8 ||
        group_score >= 15 ||
        leading_score >= 8 ||
        letter_score > 32 ||
        number_score >= 40 ||
        0
        ) {
        return SCORE_ACCEPTED;
    }
    return SCORE_REJECTED;
}
