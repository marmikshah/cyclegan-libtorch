#ifndef ARTIUM_CYCLEGAN_HPP
#define ARTIUM_CYCLEGAN_HPP

#include "blocks.hpp"
#include "globals.hpp"

namespace CycleGAN {

  using namespace torch::nn;

  struct Discriminator : Module {
    Sequential block{nullptr};
    Discriminator(int inChannels = 3, int ndf = 64, int numLayers = 3) {
      /**
       * Initialize Discriminator Network
       *
       * @param inChannels Number of channels in the input images
       * @param ndf Number of filters in the last conv layer
       * @param numLayers Number of conv layers in the network
       */

      Sequential layers = Sequential();

      torch::ExpandingArray<2UL> kernel(4);
      int padding = 1;

      layers->push_back(Conv2d(Conv2dOptions(inChannels, ndf, kernel).stride(2)));
      layers->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2)));

      int multiplier = 1, multiplierPrev = 1;
      for (int i = 1; i <= numLayers; i++) {
        multiplierPrev = multiplier;
        multiplier = min(1 << i, 8);
        layers->push_back(Conv2d(Conv2dOptions(ndf * multiplierPrev, ndf * multiplier, kernel).stride(2).padding(2)));
        layers->push_back(InstanceNorm2d(ndf * multiplier));
        layers->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));
      }

      layers->push_back(Conv2d(Conv2dOptions(ndf * multiplier, 1, kernel).stride(1).padding(padding)));

      block = register_module("block", layers);
      block->to(device);
    }
    torch::Tensor forward(torch::Tensor input) { return block->forward(input); }
  };

  struct Generator : Module {
    Sequential layers{nullptr};

    Generator(int inChannels, int outChannels, int ngf = 64, int numBlocks = 6, bool useBias = false) {
      /**
       * Initialise a Generater Network.
       * @param inChannels Number of channels of input image (3 for Color Image and 1 for Grayscale)
       * @param outChannles Number of channels of outpit image (3 for Color Image and 1 for Grayscale)
       * @param nfg Number of filters in the last convolution layer
       * @param numBlocks Number of Residual blocks in the network.
       */

      Sequential network = Sequential();

#pragma region encoder

      network->push_back(ReflectionPad2d(3));
      network->push_back(Conv2d(Conv2dOptions(inChannels, ngf, 7).padding(0).bias(useBias)));
      network->push_back(InstanceNorm2d(ngf));
      network->push_back(ReLU(true));

      int totalDownsamplingLayers = 2;
      int multiplier = 0;
      for (int i = 0; i < totalDownsamplingLayers; i++) {
        multiplier = 1 << i;
        int inFeatures = ngf * multiplier, outFeatures = ngf * multiplier * 2;
        network->push_back(Conv2d(Conv2dOptions(inFeatures, outFeatures, 3).stride(2).padding(1).bias(useBias)));
        network->push_back(InstanceNorm2d(outFeatures));
        network->push_back(ReLU(true));
      }

#pragma endregion

#pragma region transformer

      multiplier = 1 << totalDownsamplingLayers;
      for (int i = 0; i < numBlocks; i++) {
        ResidualBlock block(ngf * multiplier);
        network->push_back(block);
      }

#pragma endregion

#pragma region decoder
      // Upsampling layers
      for (int i = 0; i < totalDownsamplingLayers; i++) {
        multiplier = 1 << (totalDownsamplingLayers - i);
        int inFeatures = ngf * multiplier, outFeatures = ngf * multiplier / 2;
        network->push_back(ConvTranspose2d(ConvTranspose2dOptions(inFeatures, outFeatures, 3).stride(2).padding(1).output_padding(1).bias(useBias)));
        network->push_back(InstanceNorm2d(outFeatures));
        network->push_back(ReLU(true));
      }

      network->push_back(ReflectionPad2d(3));
      network->push_back(Conv2d(Conv2dOptions(ngf, outChannels, 7).padding(0)));
      network->push_back(Tanh());
#pragma endregion

      layers = register_module("layers", network);
      layers->to(device);
    }

    torch::Tensor forward(torch::Tensor x) { return layers->forward(x); }
  };

};

#endif