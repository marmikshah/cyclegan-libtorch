#ifndef ARTIUM_DATASET_HPP
#define ARTIUM_DATASET_HPP

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "globals.hpp"

namespace fs = std::filesystem;

void normalize(torch::Tensor& tensor) {
  torch::Tensor mean = torch::ones({3, 256, 256}) * 0.5;
  torch::Tensor std = torch::ones({3, 256, 256}) * 0.5;

  tensor = tensor / 255.0;
  tensor = (tensor - mean) / std;
}

class ImageDataset : public torch::data::datasets::Dataset<ImageDataset> {
 private:
  std::vector<std::string> images;

 public:
  explicit ImageDataset(std::string path) {
    for (const auto& entry : fs::directory_iterator(path)) {
      images.push_back(entry.path().string());
    }
  }

  torch::data::Example<> get(size_t index) override {
    cv::Mat image = cv::imread(images[index]);
    cv::resize(image, image, cv::Size(256, 256), 0, 0, 1);

    torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, at::kByte);

    tensor = tensor.permute({2, 0, 1});
    normalize(tensor);
    torch::Tensor label = torch::full({1}, 1);

    return {tensor.to(device), label.to(device)};
  }

  torch::optional<size_t> size() const override { return images.size(); }
};

#endif