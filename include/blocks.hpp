#ifndef ARTIUM_BLOCKS_HPP
#define ARTIUM_BLOCKS_HPP

#include "globals.hpp"

struct ResidualBlock : torch::nn::Module {
  torch::nn::Sequential conv{nullptr};
  torch::nn::InstanceNorm2d norm{nullptr};

  ResidualBlock(int features, bool useDropout = false) {
    torch::nn::Sequential block = torch::nn::Sequential();

    torch::ExpandingArray<2UL> kernel(3);

    block->push_back(torch::nn::ReflectionPad2d(1));

    block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(features, features, kernel)));
    block->push_back(torch::nn::InstanceNorm2d(features));
    block->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));

    if (useDropout) block->push_back(torch::nn::Dropout(0.5));

    block->push_back(torch::nn::ReflectionPad2d(1));
    block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(features, features, kernel)));
    block->push_back(torch::nn::InstanceNorm2d(features));

    conv = register_module("conv", block);
    conv->to(device);
  }
  torch::Tensor forward(torch::Tensor x) { return x + this->conv->forward(x); }
};

#endif