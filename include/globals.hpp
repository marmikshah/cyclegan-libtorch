#ifndef ARTIUM_UTILS_HPP
#define ARTIUM_UTILS_HPP

#include <torch/torch.h>

inline torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

#endif