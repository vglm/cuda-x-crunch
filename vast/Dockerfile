FROM vastai/base-image:cuda-12.6.3-cudnn-devel-ubuntu22.04-py313

# Vast need this command to work properly
WORKDIR /opt/workspace-internal/
RUN . /venv/main/bin/activate

# Our custom installation

RUN curl -sSL https://github.com/vglm/cuda-x-crunch/releases/download/v0.5.2/profanity_cuda-12.4.1 -o profanity_cuda \
    && chmod +x profanity_cuda  \
    && mv profanity_cuda /usr/local/bin/profanity_cuda

COPY runner .
CMD ["python", "runner/runner.py"]