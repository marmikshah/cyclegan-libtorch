#ifndef ARTIUM_DATASET_HPP
#define ARTIUM_DATASET_HPP

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "globals.hpp"
#include "imagetools.hpp"

namespace fs = std::filesystem;

void normalize(torch::Tensor& tensor) {
  torch::Tensor mean = torch::ones({3, 256, 256}) * 0.5;
  torch::Tensor std = torch::ones({3, 256, 256}) * 0.5;

  tensor = tensor / 255.0;
  tensor = (tensor - mean) / std;
}

struct Batch {
  torch::Tensor imagesA;
  torch::Tensor imagesB;
};

class ImageDataset {
 private:
  std::vector<std::string> imagesA;
  std::vector<std::string> imagesB;
  cv::Size dims;
  int width, height;
  bool iterationComplete = false;
  int currentIndex = 0;
  int batchSize;

 public:
  ImageDataset(std::string pathImagesA, std::string pathImagesB, int width, int height, int batchSize) {
    for (const auto& entry : fs::directory_iterator(pathImagesA)) {
      imagesA.push_back(entry.path().string());
    }
    std::cout << "Found " << imagesA.size() << " images for Category A" << std::endl;
    for (const auto& entry : fs::directory_iterator(pathImagesB)) {
      imagesB.push_back(entry.path().string());
    }
    std::cout << "Found " << imagesB.size() << " images for Category B" << std::endl;
    this->dims = cv::Size(width, height);
    this->width = width;
    this->height = height;
    this->batchSize = batchSize;
  }

  Batch* getBatch() {
    Batch* batch = new Batch();
    int maxSamplesInBatch = min(imagesA.size() - currentIndex, batchSize);
    batch->imagesA = torch::zeros({maxSamplesInBatch, 3, this->height, this->width});
    batch->imagesB = torch::zeros({maxSamplesInBatch, 3, this->height, this->width});
    for (int i = 0; i < batchSize; i++, currentIndex++) {
      if (i >= imagesA.size()) {
        iterationComplete = true;
        break;
      }
      batch->imagesA[i] = matToTensor(this->imagesA[currentIndex], this->dims);

      int randomIndex = rand() % imagesB.size();
      batch->imagesB[i] = matToTensor(this->imagesB[randomIndex], this->dims);
    }
    return batch;
  }

  void reset() {
    this->currentIndex = -1;
    this->iterationComplete = false;
  }
  bool isIterationComplete() { return this->iterationComplete; }
};

#endif