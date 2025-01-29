FROM ubuntu:22.04
RUN apt-get update
RUN apt-get -y install wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-8
RUN rm cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN apt-get -y install build-essential cmake
COPY . /app
WORKDIR /app
RUN apt-get -y install git
RUN git clean -fdx
RUN cmake .
RUN make