# GANs in LibTorch.

**:bangbang: According to [this](https://discuss.pytorch.org/t/pytorch-2-and-the-c-interface/168034/6) thread, C++ as a deployment mechanism will be fully supported, but not recommended to use for Training.**

This repository contains a collection of GAN Architectures implemented in Libtorch (C++).  
Training is done in Libtorch (C++) but, inference functions will be available in both Python and C++.  
Please note that there can be accuracy degragration when loading a model in Python.

## Setup

### Docker Environment

The repo provides to environments, one for python (pytorch) and one for C++ (libtorch).  
Python env will also setup a Jupyter server @ port 8888.
To build both enviroments, run the following command.   

```bash
docker-compose up --build
```

### Compile Training Binary

Create an executable file of the C++ library.
```bash
docker exec -it libtorch bash
mkdir build
cd build
cmake ..

make -j
```
### Training

```bash
./artium --dataset path/to/dataset --trainer dcgan|cyclegan --width 64|256 --height 64|256 --batch-size 2
```


## DCGAN 

A simple DCGAN Implementation is provided [here](./include/dcgan.hpp)  


### Example (Random Face Generation)
Results on celebA dataset (todo)

### Limitations
- Input size is limited to 64 x 64.

## CycleGAN

CycleGAN Implementation is provided [here](./include/cyclegan.hpp)
### Example (Style Transfer)
Results on monet2picture dataset (todo)

### Limitations
- Input size is limited to 256 x 256. 

