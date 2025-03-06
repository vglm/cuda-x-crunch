#pragma once
#include "create3.h"

#define SCORE_ACCEPTED 1
#define SCORE_REJECTED 0

__device__ inline uint32_t scorer(ethaddress& addr)
{
    uint8_t let_full[40];
    for (int i = 0; i < 20; i++) {
        let_full[2 * i] = (addr.b[i] >> 4) & 0x0f;
        let_full[2 * i + 1] = addr.b[i] & 0x0f;
    }

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



    if (number == __nv_bswap32(0xbadbabe0)
        || number == __nv_bswap32(0xb00bbabe)
        || number == __nv_bswap32(0xc0ffee00)
        || number == __nv_bswap32(0xdaedbeef)
        || number == __nv_bswap32(0x31415926)

        ) {
        pattern = 1;
    }


    if (
        pattern >= 1 ||
        patternbaba >= 2 ||
        pattern_score >= 3 ||
        etherscan_score >= 8 ||
        group_score >= 15 ||
        leading_score >= 8 ||
        letter_score > 32 ||
        number_score >= 40
        ) {
        return SCORE_ACCEPTED;
    }
    return SCORE_REJECTED;
}
