#!/usr/bin/env bash

curl -sSL https://github.com/vglm/cuda-x-crunch/releases/download/v0.5.2/profanity_cuda-12.4.1 -o profanity_cuda && chmod +x profanity_cuda
sudo mv profanity_cuda /usr/local/bin/profanity_cuda