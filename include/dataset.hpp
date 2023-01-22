#ifndef ARTIUM_DATASET_HPP
#define ARTIUM_DATASET_HPP

#include <torchvision/vision.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "globals.hpp"

namespace fs = std::filesystem;

torch::Tensor normalize_tensor(torch::Tensor tensor) {
  auto mean1 = torch::tensor({0.5}).repeat({256, 256});
  auto mean2 = torch::tensor({0.5}).repeat({256, 256});
  auto mean3 = torch::tensor({0.5}).repeat({256, 256});
  auto mean = torch::stack({mean1, mean2, mean3});

  auto std1 = torch::tensor({0.5}).repeat({256, 256});
  auto std2 = torch::tensor({0.5}).repeat({256, 256});
  auto std3 = torch::tensor({0.5}).repeat({256, 256});
  auto std = torch::stack({std1, std2, std3});

  tensor = tensor / 255.0;
  tensor = (tensor - mean) / std;
  return tensor;
}

class ImageDataset : public torch::data::datasets::Dataset<ImageDataset> {
 private:
  std::vector<std::string> images;

 public:
  explicit ImageDataset() {
    // TODO: Get path from command line
    std::string path = "../data/";

    for (const auto &entry : fs::directory_iterator(path)) {
      images.push_back(entry.path().string());
    }
    std::cout << "Found " << images.size() << " images";
  }

  torch::data::Example<> get(size_t index) override {
    cv::Mat image = cv::imread(images[index]);
    cv::resize(image, image, cv::Size(256, 256), 0, 0, 1);

    torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, at::kByte);
    tensor = tensor.permute({2, 0, 1});
    tensor = normalize_tensor(tensor);

    torch::Tensor label = torch::full({1}, 1);

    return {tensor.to(device), label.to(device)};
  }

  torch::optional<size_t> size() const override { return images.size(); }
};

#endif