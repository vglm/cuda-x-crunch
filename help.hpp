#pragma once

#include <string>

const std::string g_strHelp = R"(
usage: ./profanity_cuda [OPTIONS]

  Mandatory args:
    -z                      Seed public key to start, add it's private key
                            to the "profanity2" resulting private key.

  About:
    profanity2 is a vanity address generator for Ethereum that utilizes
    computing power from GPUs using OpenCL.

  Forked "profanity_cuda":
    Author: 1inch Network <info@1inch.io>
    Disclaimer:
      Ported 1inch implementation to CUDA and modified for addressology.ovh requirements

  Forked "profanity2":
    Author: 1inch Network <info@1inch.io>
    Disclaimer:
      This project "profanity2" was forked from the original project and
      modified to guarantee "SAFETY BY DESIGN". This means source code of
      this project doesn't require any audits, but still guarantee safe usage.

  From original "profanity":
    Author: Johan Gustafsson <profanity@johgu.se>
    Beer donations: 0x000dead000ae1c8e8ac27103e4ff65f42a4e9203
    Disclaimer:
      Always verify that a private key generated by this program corresponds to
      the public key printed by importing it to a wallet of your choice. This
      program like any software might contain bugs and it does by design cut
      corners to improve overall performance.)";
