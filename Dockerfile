FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt update && \
    apt install -y git vim python3-pip wget unzip cmake \
    libopencv-dev

RUN pip3 install tqdm opencv-python jupyter
WORKDIR /opt/libs/
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip -O libtorch.zip
RUN unzip libtorch.zip

ARG USER=marmikshah

RUN addgroup --gid 1000 ${USER}
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 ${USER}

USER ${USER}