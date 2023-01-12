#include <iostream>

#include "cyclegan.hpp"
#include "globals.hpp"

int main() {
  torch::Tensor tensor = torch::rand({1, 3, 224, 224}).to(device);

  std::shared_ptr gen1 = std::make_shared<CycleGAN::Generator>();
  gen1->to(device);

  std::shared_ptr gen2 = std::make_shared<CycleGAN::Generator>();
  gen2->to(device);

  torch::Tensor result = gen1->forward(tensor);
  std::cout << CycleGAN::generatorLoss(result) << std::endl;
}
