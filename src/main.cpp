#include <cxxopts.hpp>
#include <iostream>

#include "train.hpp"

int main(int argc, char** argv) {
  using namespace cxxopts;
  Options options("Artium", "Style transfer");

  const std::string group = "Artium";

  options.add_option(group, {"test", "Will perform inference if set to true", value<bool>()->default_value("false")});

  options.add_option(group, {"dataset", "Path to dataset", value<std::string>()});

  // Model Options
  options.add_option(group, {"width", "Model input width", value<int>()->default_value("256")});
  options.add_option(group, {"height", "Model input height", value<int>()->default_value("256")});
  options.add_option(group, {"r,blocks", "Num resnet blocks", value<int>()->default_value("6")});

  // Training Options
  options.add_option(group, {"e,epochs", "Total epochs for training", value<int>()->default_value("500")});
  options.add_option(group, {"b,batch-size", "Batch size", value<int>()->default_value("4")});
  options.add_option(group, {"l,learning-rate", "Learning rate", value<double>()->default_value("0.0005")});
  options.add_option(group, {"lambda-identity", "Identity loss", value<double>()->default_value("0.5")});
  options.add_option(group, {"lambda-a", "Identity loss A", value<double>()->default_value("10.0")});
  options.add_option(group, {"lambda-b", "Identity loss b", value<double>()->default_value("10.0")});

  // Export Options
  options.add_option(group, {"export-dir", "Export training results to", value<std::string>()->default_value("./")});

  auto result = options.parse(argc, argv);
  train(result);
}
