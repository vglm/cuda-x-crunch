FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder
RUN apt update && apt install -y git cmake python3
COPY . /app
WORKDIR /app
RUN git clean -fdx
RUN export CUDA_ARCHITECTURES="61 70 75 80 86 89 90" && python3 patch_cuda_architectures.py
RUN cmake . -DCMAKE_BUILD_TYPE=Release
RUN make

FROM ubuntu:22.04
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends parallel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/profanity_cuda /usr/local/bin/profanity_cuda

