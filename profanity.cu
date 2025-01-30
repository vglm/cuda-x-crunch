/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "create3.h"
#include "help.hpp"
#include "utils.hpp"
#include "ArgParser.hpp"
#include "debug_utils.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "precomp.hpp"
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <csignal>
#include "Logger.hpp"
#include "version.h"

bool g_exiting = false;
// Signal handler function
void signalHandler(int signal) {
	if (signal == SIGINT) {
		std::cout << "\nCtrl + C detected. Exiting gracefully..." << std::endl;
		g_exiting = true;
	}
}

int main(int argc, char ** argv)
{
	std::signal(SIGINT, signalHandler);

    ArgParser argp(argc, argv);
    bool bHelp = false;
    double benchmarkLimitTime = 0.0;
    int benchmarkLimitLoops = 0;
    bool bDebug = false;
    bool bErrorsOnly = false;
    bool bVersion = false;
    uint64_t uSeed = 0;

    int kernelSize = 256;
    int groups = 1000;
    int rounds = 1000;
    std::string strOutputDirectory = "output";
    std::string factoryAddr = "0x9E3F8eaE49E442A323EF2094f277Bf62752E6995";

    argp.addSwitch('s', "seed", uSeed);
    argp.addSwitch('d', "debug", bDebug);
    argp.addSwitch('v', "version", bVersion);
    argp.addSwitch('e', "errors", bErrorsOnly);
    argp.addSwitch('h', "help", bHelp);
    argp.addSwitch('b', "benchmark", benchmarkLimitTime);
    argp.addSwitch('l', "loops", benchmarkLimitLoops);
    argp.addSwitch('o', "output", strOutputDirectory);
    argp.addSwitch('f', "factory", factoryAddr);
    argp.addSwitch('k', "kernel", kernelSize);
    argp.addSwitch('g', "groups", groups);
    argp.addSwitch('r', "rounds", rounds);

    if (!argp.parse()) {
        std::cout << "error: bad arguments, -h for help" << std::endl;
        return 1;
    }
    if (uSeed) {
        LOG_WARNING("Using custom seed %llx so results will be the same and not random", uSeed);
        init_random(uSeed);
    }
    if (bVersion) {
        std::cout << g_strVersion;
        return 0;
    }
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);

            std::cout << "Device " << device << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
            std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
            std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "  Warp size: " << prop.warpSize << std::endl;
            std::cout << "  Max grid dimensions: ("
                      << prop.maxGridSize[0] << ", "
                      << prop.maxGridSize[1] << ", "
                      << prop.maxGridSize[2] << ")" << std::endl;
            std::cout << "  Max block dimensions: ("
                      << prop.maxThreadsDim[0] << ", "
                      << prop.maxThreadsDim[1] << ", "
                      << prop.maxThreadsDim[2] << ")" << std::endl;
            std::cout << std::endl;
        }
    }


    if (bDebug && bErrorsOnly) {
        std::cout << "error: can't use -d and -e together" << std::endl;
        return 1;
    }
    if (bDebug) {
        Logger::getInstance().setLevel(Logger::Level::DEBUG);
    }
    if (bErrorsOnly) {
        Logger::getInstance().setLevel(Logger::Level::ERROR);
    }

    if (bHelp) {
        std::cout << "Version: " << g_strVersion << "\n" << g_strHelp << std::endl;
        return 0;
    }
    if (benchmarkLimitLoops > 0 || benchmarkLimitTime > 0) {
        LOG_WARNING("Benchmark mode enabled");
    }
    //normalize address
    factoryAddr = normalize_ethereum_address(factoryAddr);
    if (factoryAddr.empty()) {
        std::cout << "error: bad factory address" << std::endl;
        return 1;
    }

	create3_search_data init_data = { 0 };
    memcpy(init_data.factory, factoryAddr.c_str(), 40);
    if (strOutputDirectory.size() > 1000) {
        std::cout << "error: output directory too long" << std::endl;
        return 1;
    }
    memcpy(init_data.outputDir, strOutputDirectory.c_str(), strOutputDirectory.size());
    init_data.rounds = rounds;
    init_data.kernel_group_size = kernelSize;
    init_data.kernel_groups = groups;

    LOG_INFO("Initializing with params:");
    LOG_INFO("Factory address: 0x%s", init_data.factory);
    LOG_INFO("Output directory: %s", init_data.outputDir);
    LOG_INFO("Kernel size: %d", init_data.kernel_group_size);
    LOG_INFO("Groups: %d", init_data.kernel_groups);

	create3_data_init(&init_data);
	LOG_INFO("Successfully initialised: Hashes at one run %.2f MH", (double)(init_data.kernel_groups * init_data.kernel_group_size * init_data.rounds) / 1000000.0);

    double start = get_app_time_sec();
    uint64_t loop_no = 0;
    while(true) {
		if (g_exiting) {
			break;
		}
        create3_search(&init_data);
        double end = get_app_time_sec();
        if ((benchmarkLimitTime > 0 && (end - start) > benchmarkLimitTime) || (benchmarkLimitLoops > 0 && init_data.loops + 1 > benchmarkLimitLoops)) {
            break;
        }
        loop_no += 1;
    }
    create3_data_destroy(&init_data);
    return 0;

}
