FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder
RUN apt update && apt install -y git cmake
COPY . /app
WORKDIR /app
RUN git clean -fdx
RUN cmake .
RUN make

FROM ubuntu:22.04
COPY --from=builder /app/profanity_cuda /usr/local/bin/profanity_cuda

