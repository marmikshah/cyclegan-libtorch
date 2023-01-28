#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include <cxxopts.hpp>

struct TrainingOpts {
  int numBlocks;
  int learningRate;
  int width;
  int height;
  int batchSize;
  int maxEpochs;
  std::string exportDir;
  double lambdaIdt;
  double lambdaA;
  double lambdaB;
  double stepSize;

  std::string datasetDir;
  std::string datasetA;
  std::string datasetB;

  TrainingOpts(cxxopts::ParseResult opts) {
    numBlocks = opts["blocks"].as<int>();
    learningRate = opts["learning-rate"].as<double>();
    width = opts["width"].as<int>();
    height = opts["height"].as<int>();
    batchSize = opts["batch-size"].as<int>();
    exportDir = opts["export-dir"].as<std::string>();
    lambdaIdt = opts["lambda-identity"].as<double>();
    lambdaA = opts["lambda-a"].as<double>();
    lambdaB = opts["lambda-b"].as<double>();
    datasetDir = opts["dataset"].as<std::string>();
    datasetA = datasetDir + "/trainA";
    datasetB = datasetDir + "/trainB";
    maxEpochs = opts["epochs"].as<int>();
    stepSize = opts["step-size"].as<double>();
  }
};

#endif