#ifndef ARTIUM_UTILS_HPP
#define ARTIUM_UTILS_HPP

#include <torch/torch.h>

inline torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

void setGrad(torch::nn::Sequential& model, bool grad) {
  for (auto& param : model->parameters()) {
    param.set_requires_grad(grad);
  }
}

#endif
