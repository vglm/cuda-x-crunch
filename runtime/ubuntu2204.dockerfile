FROM ubuntu:22.04
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends parallel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY profanity_cuda /usr/local/bin/profanity_cuda
RUN chmod +x /usr/local/bin/profanity_cuda
CMD ["/usr/local/bin/profanity_cuda"]