#ifndef ARTIUM_CYCLEGAN_HPP
#define ARTIUM_CYCLEGAN_HPP

#include "globals.hpp"

namespace CycleGAN {

  using namespace torch::nn;

  torch::Tensor discriminatorLoss(torch::Tensor real, torch::Tensor fake) {
    return torch::mean((real - 1) * (real - 1)) + torch::mean(fake * fake);
  }

  struct Discriminator : Module {
    Sequential block{nullptr};
    Discriminator(int nc = 3, int ndf = 64) {
      Sequential layers = Sequential();
      layers->push_back(Conv2d(getConv2dOptions(nc, ndf, 4, 2, 1, false)));
      layers->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      layers->push_back(Conv2d(getConv2dOptions(ndf, ndf * 2, 4, 2, 1, false)));
      layers->push_back(InstanceNorm2d(ndf * 2));
      layers->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      layers->push_back(Conv2d(getConv2dOptions(ndf * 2, ndf * 4, 4, 2, 1, false)));
      layers->push_back(InstanceNorm2d(0.2));
      layers->push_back(Conv2d(getConv2dOptions(ndf * 4, ndf * 8, 4, 1, 1)));
      layers->push_back(InstanceNorm2d(ndf * 8));
      layers->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));
      layers->push_back(Conv2d(getConv2dOptions(ndf * 8, 1, 4, 1, 1)));

      block = register_module("block", layers);
      block->to(device);
    }
    torch::Tensor forward(torch::Tensor input) { return block->forward(input); }
  };

  torch::Tensor generatorLoss(torch::Tensor fake) { return torch::mean((fake - 1) * (fake - 1)); }

  struct ResidualBlock : Module {
    Sequential conv{nullptr};
    InstanceNorm2d norm{nullptr};

    ResidualBlock(int features) {
      Conv2dOptions options = getConv2dOptions(features, features, 3, 1, 1);

      Sequential block = Sequential();
      block->push_back(Conv2d(options));
      block->push_back(InstanceNorm2d(features));
      block->push_back(ReLU());
      block->push_back(Conv2d(options));

      conv = register_module("conv", block);
      conv->to(device);
      norm = register_module("norm", InstanceNorm2d(features));
      norm->to(device);
    }

    torch::Tensor forward(torch::Tensor x) { return torch::relu(this->norm->forward(this->conv->forward(x) + x)); }
  };

  struct Generator : Module {
    Sequential layers{nullptr};

    Generator(int features = 64, int blocks = 6) {
      Sequential network = Sequential();

      network->push_back(ReflectionPad2d(3));
      network->push_back(Conv2d(getConv2dOptions(3, features, 7, 1, 0)));
      network->push_back(InstanceNorm2d(2));
      network->push_back(ReLU(true));
      network->push_back(Conv2d(getConv2dOptions(features, 2 * features, 3, 2, 1)));
      network->push_back(InstanceNorm2d(2 * features));
      network->push_back(ReLU(true));
      network->push_back(Conv2d(getConv2dOptions(2 * features, 4 * features, 3, 2, 1)));
      network->push_back(InstanceNorm2d(2));
      network->push_back(ReLU(true));

      for (int i = 0; i < blocks; ++i) {
        ResidualBlock block(4 * features);
        network->push_back(block);
      }

      network->push_back(createConvTranspose2d(4 * features, 4 * 2 * features, 3, 1, 1));
      network->push_back(PixelShuffle(2));
      network->push_back(InstanceNorm2d(2 * features));
      network->push_back(ReLU(true));

      network->push_back(createConvTranspose2d(2 * features, 4 * features, 3, 1, 1));
      network->push_back(PixelShuffle(2));
      network->push_back(InstanceNorm2d(2 * features));
      network->push_back(ReLU(true));

      network->push_back(ReflectionPad2d(3));
      network->push_back(Conv2d(getConv2dOptions(features, 3, 7, 1, 0)));
      network->push_back(Tanh());

      layers = register_module("layers", network);
      layers->to(device);
    }

    torch::Tensor forward(torch::Tensor x) { return layers->forward(x); }
  };

};

#endif