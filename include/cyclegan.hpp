#ifndef ARTIUM_CYCLEGAN_HPP
#define ARTIUM_CYCLEGAN_HPP

#include "blocks.hpp"
#include "globals.hpp"

namespace CycleGAN {

  using namespace torch::nn;
  void initWeights(torch::nn::Module &module) {
  std:
    // torch::NoGradGuard noGrad;
    if (auto *layer = module.as<torch::nn::Conv2dImpl>()) {
      torch::nn::init::xavier_normal_(layer->weight, 0.2);
      torch::nn::init::constant_(layer->bias, 0.0);
    }

    if (auto *layer = module.as<torch::nn::BatchNorm2dImpl>()) {
      torch::nn::init::normal_(layer->weight, 1.0, 0.2);
      torch::nn::init::constant_(layer->bias, 0.0);
    }
  }


  struct Discriminator : Module {
    Sequential block;
    Discriminator(int inChannels = 3, int ndf = 64, int numLayers = 3) {
      /**
       * Initialize Discriminator Network
       *
       * @param inChannels Number of channels in the input images
       * @param ndf Number of filters in the last conv layer
       * @param numLayers Number of conv layers in the network
       */


      torch::ExpandingArray<2UL> kernel(4);
      int padding = 1;

      block->push_back(Conv2d(Conv2dOptions(inChannels, ndf, kernel).stride(2)));
      block->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2)));

      int multiplier = 1, multiplierPrev = 1;
      for (int i = 1; i <= numLayers; i++) {
        multiplierPrev = multiplier;
        multiplier = min(1 << i, 8);
        block->push_back(Conv2d(Conv2dOptions(ndf * multiplierPrev, ndf * multiplier, kernel).stride(2).padding(2)));
        block->push_back(BatchNorm2d(ndf * multiplier));
        block->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));
      }

      block->push_back(Conv2d(Conv2dOptions(ndf * multiplier, 1, kernel).stride(1).padding(padding)));

      register_module("block", block);
      block->apply(initWeights);
    }
    torch::Tensor forward(torch::Tensor input) { return this->block->forward(input); }
  };

  struct Generator : Module {
    Sequential layers;

    Generator(int inChannels, int outChannels, int ngf = 64, int numBlocks = 6, bool useBias = true) {
      /**
       * Initialise a Generater Network.
       * @param inChannels Number of channels of input image (3 for Color Image and 1 for Grayscale)
       * @param outChannles Number of channels of outpit image (3 for Color Image and 1 for Grayscale)
       * @param nfg Number of filters in the last convolution layer
       * @param numBlocks Number of Residual blocks in the network.
       */

#pragma region encoder

      layers->push_back(ReflectionPad2d(3));
      layers->push_back(Conv2d(Conv2dOptions(inChannels, ngf, 7).padding(0).bias(useBias)));
      layers->push_back(BatchNorm2d(ngf));
      layers->push_back(ReLU(true));

      int totalDownsamplingLayers = 2;
      int multiplier = 0;
      for (int i = 0; i < totalDownsamplingLayers; i++) {
        multiplier = 1 << i;
        int inFeatures = ngf * multiplier, outFeatures = ngf * multiplier * 2;
        layers->push_back(Conv2d(Conv2dOptions(inFeatures, outFeatures, 3).stride(2).padding(1).bias(useBias)));
        layers->push_back(BatchNorm2d(outFeatures));
        layers->push_back(ReLU(true));
      }

#pragma endregion

#pragma region transformer

      multiplier = 1 << totalDownsamplingLayers;
      for (int i = 0; i < numBlocks; i++) {
        ResidualBlock block(ngf * multiplier);
        layers->push_back(block);
      }

#pragma endregion

#pragma region decoder
      // Upsampling layers
      for (int i = 0; i < totalDownsamplingLayers; i++) {
        multiplier = 1 << (totalDownsamplingLayers - i);
        int inFeatures = ngf * multiplier, outFeatures = ngf * multiplier / 2;
        layers->push_back(ConvTranspose2d(
            ConvTranspose2dOptions(inFeatures, outFeatures, 3).stride(2).padding(1).output_padding(1).bias(useBias)));
        layers->push_back(BatchNorm2d(outFeatures));
        layers->push_back(ReLU(true));
      }

      layers->push_back(ReflectionPad2d(3));
      layers->push_back(Conv2d(Conv2dOptions(ngf, outChannels, 7).padding(0)));
      layers->push_back(Tanh());
#pragma endregion

      register_module("layers", layers);
      layers->apply(initWeights);
    }

    torch::Tensor forward(torch::Tensor x) { return layers->forward(x); }
  };

};

#endif