#ifndef ARTIUM_DATASET_HPP
#define ARTIUM_DATASET_HPP

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "globals.hpp"
#include "imagetools.hpp"

namespace fs = std::filesystem;

class ImageDataset : public torch::data::datasets::Dataset<ImageDataset> {
  /**
   * Custom Dataset class to create a tensor for both Domain images.
   * This Dataset must have the <Stack> transform when creating it.
   */

  using Example = torch::data::Example<>;
  std::vector<std::string> pathsA;
  std::vector<std::string> pathsB;
  cv::Size dims;


 public:
  explicit ImageDataset(std::string pathA, std::string pathB, int width, int height) {
    for (const auto& entry : fs::directory_iterator(pathA)) {
      pathsA.push_back(entry.path().string());
    }
    for (const auto& entry : fs::directory_iterator(pathB)) {
      pathsB.push_back(entry.path().string());
    }
    this->dims = cv::Size(height, width);
  }

  Example get(size_t index) override {
    /**
     * As C++ Frontend does not provide a way to return Map,
     * we use a dimension to indicate the domain.
     *
     * We add an extra dimention for the domain in this tensor, named D.
     * Final output would be:
     * [Domain, Channels, Height, Width].
     *
     * The dataloader would stack it into a batch creating a tensor shaped
     *
     * We will permute this tensor from ([n, d, c, h ,w]) to ([d, n, c, h ,w])
     * The `d` domain will always be of size 2.
     * Assuming d[0] = Domain A and d[1] = Domain B
     */

    torch::Tensor sampleA = matToTensor(this->pathsA[index % pathsA.size()], this->dims);
    torch::Tensor sampleB = matToTensor(this->pathsB[index], this->dims);

    return {sampleA, sampleB};
  }

  torch::optional<size_t> size() const override { return max(this->pathsA.size(), this->pathsB.size()); }
};

#endif