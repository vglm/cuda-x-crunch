FROM ubuntu:24.04
COPY profanity_cuda /usr/local/bin/profanity_cuda
CMD ["/usr/local/bin/profanity_cuda"]