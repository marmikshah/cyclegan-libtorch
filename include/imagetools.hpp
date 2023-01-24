#ifndef ARTIUM_IMAGEPOOL_HPP
#define ARTIUM_IMAGEPOOL_HPP

#include <opencv2/opencv.hpp>

#include "globals.hpp"

bool tensorToMat(torch::Tensor tensor, std::string imagePath) {
  /**
   * Convert a Tensor to OpenCV Mat and save to disk.
   *
   * @param tensor The torch::Tensor of (c, h, w) shape
   * @param imagePage Path where image should be written to
   */

  int width = tensor.sizes()[2];
  int height = tensor.sizes()[1];
  tensor = tensor.detach().permute({1, 2, 0}).contiguous();

  tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8).to(torch::kCPU);
  cv::Mat output(width, height, CV_8UC3, tensor.data_ptr<uchar>());
  return cv::imwrite(imagePath, output);
}

class ImagePool {
  /***
   * Create an ImagePool consisting of previously
   * generated images. Images from this pool will
   * be used to train the discriminator as fake
   * images.
   */
 private:
  std::vector<torch::Tensor> pool;
  int batchSize;

 public:
  ImagePool(int batchSize, int seed = 42) {
    /**
     * @param batchSize Size of the returned batch of images.
     * @param seed Optional param to set the random seed
     */
    this->batchSize = batchSize;
    std::cout << "Created ImagePool with seed (" << seed << ")" << std::endl;
    srand(seed);
  }

  torch::Tensor getImages(torch::Tensor images) {
    /**
     * Create a batch of generated images.
     * Selection process will take images from this->pool
     * or @param images.
     *
     * @param images: Tensor containing currently generated images.
     */
    torch::Tensor batch = torch::zeros({this->batchSize, 3, 256, 256});

    int totalImages = 0;
    for (int i = 0; i < images.sizes()[0]; i++) {
      if (totalImages < this->batchSize) {
        totalImages += 1;
        batch[i] = images[i];
        pool.push_back(images[i]);
      } else {
        if (((double)rand() / (RAND_MAX)) + 1 > 0.5) {
          int randomIndex = rand() % pool.size();
          batch[i] = pool[randomIndex];
          pool[randomIndex] = images[i];
          totalImages += 1;
        } else {
          batch[i] = images[i];
        }
      }
    }
    return batch;
  }
};

#endif