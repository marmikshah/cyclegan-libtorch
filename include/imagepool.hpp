#ifndef ARTIUM_IMAGEPOOL_HPP
#define ARTIUM_IMAGEPOOL_HPP

#include "globals.hpp"

class ImagePool {
 private:
  std::vector<torch::Tensor> pool;
  int size;

 public:
  ImagePool(int size) { this->size = size; }

  torch::Tensor getImages(torch::Tensor images) {
    torch::Tensor output = torch::zeros({this->size, 3, 256, 256});
    int totalImages = 0;
    for (int i = 0; i < images.sizes()[0]; i++) {
      if (totalImages < this->size) {
        totalImages += 1;
        output[i] = images[i];
        pool.push_back(images[i]);
      } else {
        if (((double)rand() / (RAND_MAX)) + 1 > 0.5) {
          int randomIndex = rand() % pool.size();
          output[i] = pool[randomIndex];
          pool[randomIndex] = images[i];
        } else {
          output[i] = images[i];
        }
      }
    }
    return output;
  }
};

#endif