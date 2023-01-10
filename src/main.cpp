#include <torch/torch.h>

#include <iostream>

#include "generator.hpp"
#include "globals.hpp"

int main() {
  torch::Tensor tensor = torch::rand({1, 3, 224, 224}).to(device);

  std::shared_ptr gen1 = std::make_shared<Generator>();
  gen1->to(device);
  std::cout << gen1->forward(tensor) << std::endl;
}
