#ifndef ARTIUM_STRUCTS_HPP
#define ARTIUM_STRUCTS_HPP

#include <cxxopts.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct SettingsBase {
  /**
   * Command Line arugments parser.
  */
  cxxopts::ParseResult opts;

  SettingsBase(cxxopts::ParseResult opts) { this->opts = opts; }

  /* Path Configurations*/
  std::string getDatasetDirectory() {
    std::string datasetDir = opts["dataset"].as<std::string>();

    if (!fs::exists(fs::path(datasetDir))) {
      std::cout << "Directory " << datasetDir << " does not exist" << std::endl;
      exit(-1);
    }
    return datasetDir;
  }

  std::string getExperimentDirectory() {
    std::string experimentDir = opts["export-dir"].as<std::string>();
    fs::create_directory(fs::path(experimentDir));
    return experimentDir;
  }

  std::string getPreviewsDirectory() {
    std::string previewDir = getExperimentDirectory() + "/previews/";
    fs::create_directory(fs::path(previewDir));
    return previewDir;
  }

  /* Image Dimensions */
  int getInputWidth() { return opts["width"].as<int>(); }
  int getInputHeight() { return opts["height"].as<int>(); }
  cv::Size getInputSize() { return cv::Size(getInputHeight(), getInputWidth()); }

  /* Training Configurations */
  double getLearningRate() { return opts["learning-rate"].as<double>(); }
  int getTotalEpochs() { return opts["epochs"].as<int>(); }
  int getBatchSize() { return opts["batch-size"].as<int>(); }
  int getStepInterval() { return opts["step-interval"].as<int>(); }
  int getStepSize() { return opts["step-size"].as<int>(); }
};

#endif