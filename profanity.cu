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
    bool bModeBenchmark = false;
    bool bDebug = false;
    bool bErrorsOnly = false;

    int kernelSize = 256;
    int groups = 1000;
    int rounds = 1000;
    std::string strOutputDirectory = "output";
    std::string factoryAddr = "0x9E3F8eaE49E442A323EF2094f277Bf62752E6995";

    argp.addSwitch('d', "debug", bDebug);
    argp.addSwitch('e', "errors", bErrorsOnly);
    argp.addSwitch('h', "help", bHelp);
    argp.addSwitch('b', "benchmark", bModeBenchmark);
    argp.addSwitch('o', "output", strOutputDirectory);
    argp.addSwitch('f', "factory", factoryAddr);
    argp.addSwitch('k', "kernel", kernelSize);
    argp.addSwitch('g', "groups", groups);
    argp.addSwitch('r', "rounds", rounds);

    if (!argp.parse()) {
        std::cout << "error: bad arguments, -h for help" << std::endl;
        return 1;
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
        std::cout << g_strHelp << std::endl;
        return 0;
    }
    if (bModeBenchmark) {
        LOG_INFO("Benchmark mode enabled - application will run for 10 seconds");
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
    while(true) {
		if (g_exiting) {
			break;
		}
        create3_search(&init_data);
        double end = get_app_time_sec();
        if (bModeBenchmark && (end - start) > 10) {
            break;
        }
    }
    create3_data_destroy(&init_data);
    return 0;

}
