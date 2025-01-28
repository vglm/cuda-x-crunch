#include "debug_utils.hpp"
#include <iostream>


void pretty_print_mp_number(const mp_number & n) {
    printf("mp_number: ");
    for (int i = 0; i < MP_WORDS; ++i) {
        printf("0x%08x ", n.d[i]);
    }
    printf("\n");
}

