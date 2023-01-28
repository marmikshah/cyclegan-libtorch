#ifndef ARTIUM_IMAGEPOOL_HPP
#define ARTIUM_IMAGEPOOL_HPP

#include <opencv2/opencv.hpp>

#include "globals.hpp"

void normalize(torch::Tensor& tensor) {
  torch::Tensor mean = torch::ones({3, 256, 256}) * 0.5;
  torch::Tensor std = torch::ones({3, 256, 256}) * 0.5;

  tensor = tensor / 255.0;
  tensor = (tensor - mean) / std;
}

cv::Mat tensorToMat(torch::Tensor tensor, bool doDenorm = false) {
  /**
   * Convert a Tensor to OpenCV Mat
   *
   * @param tensor The torch::Tensor of (c, h, w) shape
   */

  int width = tensor.sizes()[2];
  int height = tensor.sizes()[1];
  if (doDenorm) {
    tensor = tensor.add(1).div_(2).clamp_(0, 1);
  }
  tensor = tensor.mul(255).add_(0.5).clamp(0, 255).permute({1, 2, 0}).to(torch::kCPU, torch::kUInt8, false, false,
                                                                         torch::MemoryFormat::Contiguous);

  cv::Mat output(width, height, CV_8UC3, tensor.data_ptr<uchar>());
  return output;
}

torch::Tensor matToTensor(std::string& path, cv::Size& size, bool doNormalize = true, bool doPermute = true) {
  cv::Mat mat = cv::imread(path);
  cv::resize(mat, mat, size, 0, 0, 1);

  std::vector<cv::Mat> channels(3);
  cv::split(mat, channels);

  auto R = torch::from_blob(channels[2].ptr(), {size.height, size.width}, torch::kUInt8);
  auto G = torch::from_blob(channels[1].ptr(), {size.height, size.width}, torch::kUInt8);
  auto B = torch::from_blob(channels[0].ptr(), {size.height, size.width}, torch::kUInt8);

  torch::Tensor tensor = torch::cat({B, G, R}).view({3, size.height, size.width}).to(torch::kFloat);

  if (doNormalize) normalize(tensor);
  return tensor;
}

class ImagePool {
  /***
   * Create an ImagePool consisting of previously
   * generated images. Images from this pool will
   * be used to train the discriminator as fake
   * images.
   */
 private:
  std::deque<torch::Tensor> pool;
  int poolSize;

 public:
  ImagePool(int poolSize = 50) {
    /**
     * @param batchSize Size of the returned batch of images.
     */
    this->poolSize = poolSize;
  }

  torch::Tensor getImages(torch::Tensor images) {
    /**
     * Create a batch of generated images.
     * Selection process will take images from this->pool
     * or @param images.
     *
     * @param images: Tensor containing currently generated images.
     */
    std::vector<torch::Tensor> batch;

    for (int i = 0; i < images.sizes()[0]; i++) {
      if (pool.size() < this->poolSize) {
        batch.push_back(images[i].detach().clone());
        pool.push_back(images[i]);
      } else {
        if (((double)rand() / (RAND_MAX)) + 1 > 0.5) {
          int randomIndex = rand() % pool.size();
          torch::Tensor tmp = pool[randomIndex];
          pool[randomIndex] = images[i];
          batch.push_back(tmp.detach().clone());
        } else {
          batch.push_back(images[i].detach().clone());
        }
      }
    }
    return torch::stack(torch::TensorList(batch)).to(device);
  }
};

#endif