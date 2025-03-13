FROM vastai/base-image:cuda-12.6.3-cudnn-devel-ubuntu22.04-py313

# Vast need this command to work properly
WORKDIR /opt/workspace-internal/
RUN . /venv/main/bin/activate

# Our custom installation

COPY runner .
COPY profanity_cuda /usr/local/bin/profanity_cuda
CMD ["python", "-u", "runner/runner.py"]