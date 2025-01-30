FROM ghcr.io/vglm/cuda-build-base:latest as builder
COPY . /app
WORKDIR /app
RUN git clean -fdx
RUN cmake .
RUN make

FROM ubuntu:22.04
COPY --from=builder /app/profanity_cuda /usr/local/bin/profanity_cuda

