#ifndef ARTIUM_GENERATOR_HPP
#define ARTIUM_GENERATOR_HPP

#include <torch/torch.h>

#include "globals.hpp"

struct ResidualBlock : torch::nn::Module {
  torch::nn::Sequential conv{nullptr};
  torch::nn::InstanceNorm2d norm{nullptr};

  ResidualBlock(int features) {
    torch::nn::Conv2dOptions options(features, features,
                                     torch::ExpandingArray<2>(3));
    options.stride(1);
    options.padding(1);

    conv = register_module(
        "conv",
        torch::nn::Sequential(torch::nn::Conv2d(options),
                              torch::nn::InstanceNorm2d(features),
                              torch::nn::ReLU(), torch::nn::Conv2d(options)));
    conv->to(device);
    norm = register_module("norm", torch::nn::InstanceNorm2d(features));
    norm->to(device);
  }

  torch::Tensor forward(torch::Tensor x) {
    return torch::relu(this->norm->forward(this->conv->forward(x) + x));
  }
};

struct Generator : torch::nn::Module {
  ResidualBlock *block1;

  Generator() { block1 = new ResidualBlock(3); }

  torch::Tensor forward(torch::Tensor x) { return block1->forward(x); }
};

#endif
