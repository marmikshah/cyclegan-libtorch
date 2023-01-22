FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt update && \
    apt install -y git vim wget unzip cmake

RUN apt install -y libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev

WORKDIR /opt/libs/

# -------------- Build Libtorch & Torchvision -------------- #
FROM base as libtorch

ARG TORCH_VER=1.13.1
ARG TORCHVISION_VER=0.14.1

WORKDIR /opt/libs/
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-${TORCH_VER}%2Bcu117.zip -O libtorch.zip
RUN unzip libtorch.zip

RUN wget https://github.com/pytorch/vision/archive/refs/tags/v${TORCHVISION_VER}.zip -O /opt/libs/torchvision.zip 
RUN unzip torchvision.zip && mv vision-${TORCHVISION_VER} vision
WORKDIR /opt/libs/vision/build
RUN cmake -DCMAKE_PREFIX_PATH=/opt/libs/libtorch -DWITH_CUDA=yes .. \
    && make -j8

# -------------- OpenCV -------------- #
FROM base as opencv
ARG OPENCV_VER=4.7.0

WORKDIR /opt/libs/
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VER}.zip 
RUN unzip opencv.zip && mv opencv-${OPENCV_VER} opencv
WORKDIR /opt/libs/opencv/build/
RUN cmake .. && make -j8


FROM base

COPY --from=libtorch /opt/libs/libtorch/ /opt/libs/libtorch/
COPY --from=libtorch /opt/libs/vision/ /opt/libs/vision/
COPY --from=opencv /opt/libs/opencv/ /opt/libs/opencv/

RUN cd /opt/libs/vision/build/ && make install
RUN cd /opt/libs/vision/build/ && make install

ARG USER=marmikshah

RUN addgroup --gid 1000 ${USER}
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 ${USER}

USER ${USER}