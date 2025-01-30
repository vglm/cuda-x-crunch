#pragma once

#include <string>

const std::string g_strHelp = R"(
usage: ./profanity_cuda [OPTIONS]

  Args:
    -d Show debug output
    -e Show errors only
    -o Output directory (default output), it will be created if not exists
    -f Factory of create3 contract (default 0x9e3f8eae49e442a323ef2094f277bf62752e6995)
    -k Kernel group size (default 256)
    -g Number of kernel groups to run in one kernel call (default 10000)
    -r Rounds (loops) to run inside the kernel (every thread looping) (default 2000)

  Extra
    -h Show this help
    -v Show version

  Debug
    -s Seed for random number generator (default 0 means random)
    -l Limit loops for benchmark (default 0 means no limit)
    -b Limit time in seconds for benchmark (default 0 means no limit)

  About:
    profanity_cuda is a vanity address generator for (modified) contractX deployer using CUDA.

  Binary compiled from:
    https://github.com/vglm/cuda-x-crunch
)";
