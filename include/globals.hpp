#ifndef ARTIUM_UTILS_HPP
#define ARTIUM_UTILS_HPP

#include <torch/torch.h>

inline torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

inline torch::nn::Conv2dOptions getConv2dOptions(int inChannels, int outChannels, int kernel, int stride, int padding,
                                                 bool bias = true) {
  return torch::nn::Conv2dOptions(inChannels, outChannels, torch::ExpandingArray<2>(kernel))
      .stride(stride)
      .padding(padding)
      .bias(bias);
}

inline torch::nn::ConvTranspose2dOptions getConvTranspose2dOptions(int inChannels, int outChannels, int kernel,
                                                                   int stride, int padding) {
  return torch::nn::ConvTranspose2dOptions(inChannels, outChannels, torch::ExpandingArray<2>(kernel))
      .stride(stride)
      .padding(padding);
}

inline torch::nn::ConvTranspose2d createConvTranspose2d(int inChannels, int outChannels, int kernel, int stride,
                                                        int padding) {
  return torch::nn::ConvTranspose2d(getConvTranspose2dOptions(inChannels, outChannels, kernel, stride, padding));
}

#endif