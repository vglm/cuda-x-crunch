docker build -t cuda .
docker run --gpus=all cuda profanity_cuda -b 1

gvmkit-build cuda