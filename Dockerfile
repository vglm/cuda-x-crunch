FROM ghcr.io/vglm/cuda-build-base:cuda-12.8-ubuntu-22.04 as builder
COPY . /app
WORKDIR /app
RUN git clean -fdx
RUN cmake .
RUN make

FROM ubuntu:22.04
COPY --from=builder /app/profanity_cuda /usr/local/bin/profanity_cuda

