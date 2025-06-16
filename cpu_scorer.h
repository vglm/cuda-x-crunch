#pragma once
#include "create3.h"

#define SCORE_ACCEPTED 1
#define SCORE_REJECTED 0


inline uint32_t bswap32(uint32_t x) {
    return ((x & 0x000000FFU) << 24) |
           ((x & 0x0000FF00U) << 8)  |
           ((x & 0x00FF0000U) >> 8)  |
           ((x & 0xFF000000U) >> 24);
}

#define MATCH0_32(a, MASK) \
    ((a & bswap32(MASK)) == 0x0)

inline uint32_t cpu_scorer(ethaddress& addr, uint64_t extra_prefix)
{
    int group_score = 0;
    int letter_score = 0;
    int number_score = 0;


    int number_of_zeroes = 0;
    {
        uint8_t let_full[40];
        for (int i = 0; i < 20; i++) {
            let_full[2 * i] = (addr.b[i] >> 4) & 0x0f;
            let_full[2 * i + 1] = addr.b[i] & 0x0f;
        }

        for (int i = 0; i < 40; i++) {
            uint8_t letter = let_full[i];

            if (i > 0 && letter == let_full[i - 1]) {
                group_score += 1;
            }
            if (letter >= 10) {
                letter_score += 1;
            }
            if (letter < 10) {
                number_score += 1;
            }
            if (letter == 0) {
                number_of_zeroes += 1;
            }
        }
    }

    int pattern_repeats = 0;
    if (addr.d[0] == addr.d[1] && addr.d[1] == addr.d[2] && addr.d[2] == addr.d[3]) {
        pattern_repeats = 1;
    }

    int pattern = 0;
    uint32_t number = addr.d[0];
    if (number == bswap32(0xbadbabe0)
        || number == bswap32(0x01234567)
        || number == bswap32(0x12345678)
        || number == bswap32(0xb00bbabe)
        || number == bswap32(0xc0ffee00)
        || number == bswap32(0xdaedbeef)
        || number == bswap32(0xdaedf00d)
        || number == bswap32(0x31415926)
        || number == bswap32(0x00000000)
        || number == bswap32(0x11111111)
        || number == bswap32(0x22222222)
        || number == bswap32(0x33333333)
        || number == bswap32(0x44444444)
        || number == bswap32(0x55555555)
        || number == bswap32(0x66666666)
        || number == bswap32(0x77777777)
        || number == bswap32(0x88888888)
        || number == bswap32(0x99999999)
        || number == bswap32(0xaaaaaaaa)
        || number == bswap32(0xbbbbbbbb)
        || number == bswap32(0xcccccccc)
        || number == bswap32(0xdddddddd)
        || number == bswap32(0xeeeeeeee)
        || number == bswap32(0xffffffff)
        || (number & 0x00FFFFFF) == (extra_prefix & 0x00FFFFFF)
        ) {
        pattern = 1;
    }

    int pattern_zeroes = 0;


    if (
        pattern_zeroes >= 1 ||
        pattern >= 1 ||
        pattern_repeats >= 1 ||
        group_score >= 15 ||
        letter_score > 32 ||
        number_score >= 40 ||
        number_of_zeroes >= 17 ||
        0
        ) {
        return SCORE_ACCEPTED;
    }
    return SCORE_REJECTED;
}
