FROM ubuntu:22.04

RUN apt update && apt install -y python3 python3-requests python-is-python3

# Vast need this command to work properly
WORKDIR /opt/workspace-internal/

COPY runner ./runner
COPY profanity_cuda /usr/local/bin/profanity_cuda

ENTRYPOINT ["python", "-u", "runner/runner.py"]