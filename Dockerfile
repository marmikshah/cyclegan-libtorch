FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt update && \
    apt install -y git vim wget unzip cmake

RUN apt install -y libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev ffmpeg libsm6 libxext6

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
RUN cmake -DWITH_FFMPEG=on .. && make -j8


FROM base as cpp

COPY --from=libtorch /opt/libs/libtorch/ /opt/libs/libtorch/
COPY --from=libtorch /opt/libs/vision/ /opt/libs/vision/
COPY --from=opencv /opt/libs/opencv/ /opt/libs/opencv/

RUN cd /opt/libs/vision/build/ && make install
RUN cd /opt/libs/opencv/build/ && make install

RUN wget https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.0.0.zip -O /opt/libs/cxxopts.zip && \
    unzip cxxopts.zip && mv cxxopts-3.0.0 cxxopts/ && rm cxxopts.zip

ARG USER=marmikshah

RUN addgroup --gid 1000 ${USER}
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 ${USER}

USER ${USER}

FROM base as python

RUN apt install -y python3-pip
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision opencv-python pillow jupyter

RUN pip3 install jupyterthemes && jt -t onedork

ARG USER=marmikshah

RUN addgroup --gid 1000 ${USER}
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 ${USER}

USER ${USER}
