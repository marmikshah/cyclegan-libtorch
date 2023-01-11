#ifndef ARTIUM_GENERATOR_HPP
#define ARTIUM_GENERATOR_HPP

#include <torch/torch.h>

#include "globals.hpp"

inline torch::nn::Conv2dOptions getConv2dOptions(int inFeatures, int outFeatures, int kernel, int stride, int padding) {
  return torch::nn::Conv2dOptions(inFeatures, outFeatures, torch::ExpandingArray<2>(kernel))
      .stride(stride)
      .padding(padding);
}

inline torch::nn::ConvTranspose2dOptions getConvTranspose2dOptions(int inFeatures, int outFeatures, int kernel,
                                                                   int stride, int padding) {
  return torch::nn::ConvTranspose2dOptions(inFeatures, outFeatures, torch::ExpandingArray<2>(kernel))
      .stride(stride)
      .padding(padding);
}

inline torch::nn::ConvTranspose2d createConvTranspose2d(int inFeatures, int outFeatures, int kernel,
                                                                   int stride, int padding) {
  return torch::nn::ConvTranspose2d(getConvTranspose2dOptions(inFeatures, outFeatures, kernel, stride, padding));
}

struct ResidualBlock : torch::nn::Module {
  torch::nn::Sequential conv{nullptr};
  torch::nn::InstanceNorm2d norm{nullptr};

  ResidualBlock(int features) {
    torch::nn::Conv2dOptions options = getConv2dOptions(features, features, 3, 1, 1);

    torch::nn::Sequential block = torch::nn::Sequential();
    block->push_back(torch::nn::Conv2d(options));
    block->push_back(torch::nn::InstanceNorm2d(features));
    block->push_back(torch::nn::ReLU());
    block->push_back(torch::nn::Conv2d(options));

    conv = register_module("conv", block);
    conv->to(device);
    norm = register_module("norm", torch::nn::InstanceNorm2d(features));
    norm->to(device);
  }

  torch::Tensor forward(torch::Tensor x) { return torch::relu(this->norm->forward(this->conv->forward(x) + x)); }
};

struct Generator : torch::nn::Module {
  torch::nn::Sequential layers{nullptr};

  Generator(int features = 64, int blocks = 6) {
    torch::nn::Sequential network = torch::nn::Sequential();

    network->push_back(torch::nn::ReflectionPad2d(3));
    network->push_back(torch::nn::Conv2d(getConv2dOptions(3, features, 7, 1, 0)));
    network->push_back(torch::nn::InstanceNorm2d(2));
    network->push_back(torch::nn::ReLU(true));
    network->push_back(torch::nn::Conv2d(getConv2dOptions(features, 2 * features, 3, 2, 1)));
    network->push_back(torch::nn::InstanceNorm2d(2 * features));
    network->push_back(torch::nn::ReLU(true));
    network->push_back(torch::nn::Conv2d(getConv2dOptions(2 * features, 4 * features, 3, 2, 1)));
    network->push_back(torch::nn::InstanceNorm2d(2));
    network->push_back(torch::nn::ReLU(true));

    for (int i = 0; i < blocks; ++i) {
      ResidualBlock block(4 * features);
      network->push_back(block);
    }

    network->push_back(createConvTranspose2d(4 * features, 4 * 2 * features, 3, 1, 1));
    network->push_back(torch::nn::PixelShuffle(2));
    network->push_back(torch::nn::InstanceNorm2d(2 * features));
    network->push_back(torch::nn::ReLU(true));

    network->push_back(createConvTranspose2d(2 * features, 4 * features, 3, 1, 1));
    network->push_back(torch::nn::PixelShuffle(2));
    network->push_back(torch::nn::InstanceNorm2d(2 * features));
    network->push_back(torch::nn::ReLU(true));

    network->push_back(torch::nn::ReflectionPad2d(3));
    network->push_back(torch::nn::Conv2d(getConv2dOptions(features, 3, 7, 1, 0)));
    network->push_back(torch::nn::Tanh());

    layers = register_module("layers", network);
    layers->to(device);
  }

  torch::Tensor forward(torch::Tensor x) { return layers->forward(x); }
};

#endif
