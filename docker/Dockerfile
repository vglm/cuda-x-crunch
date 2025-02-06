FROM ubuntu:22.04 AS libs

FROM ubuntu:22.04
RUN rm -rf /var
RUN rm -rf /usr/share
RUN rm -rf /lib/apt  \
    && rm -rf /lib/dpkg  \
    && rm -rf /lib/systemd \
    && rm -rf /lib/locale
RUN rm -rf /usr/lib/x86_64-linux-gnu/*/
RUN find /usr/lib/x86_64-linux-gnu -type f  \
    ! -name "libc.so*"  \
    ! -name "libdl.so*"  \
    ! -name "libpthread.so*"  \
    ! -name "librt.so*"  \
    ! -name "libnsl.so*"  \
    ! -name "libtinfo.so*"  \
    ! -name "libm.so*"  \
    ! -name "libstdc*"  \
    ! -name "libgcc_s*"  \
    ! -name "ld-linux-x86-64.so*"  \
    -delete
RUN rm -rf /usr/sbing
RUN rm -rf /usr/bin

COPY profanity_cuda-12.2.1 /usr/bin/profanity_cuda
