#include <torch/torch.h>

#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3}).to(torch::Device(torch::kCUDA, 0));
  std::cout << tensor << std::endl;
}
