#include <cxxopts.hpp>
#include <iostream>

#include "dcgan.hpp"
#include "cyclegan.hpp"


int main(int argc, char** argv) {
  using namespace cxxopts;
  Options options("Artium", "Style transfer");

  const std::string group = "Artium";

  options.add_option(group, {"trainer", "Which training to do? (cyclegan, dcgan)", value<std::string>()->default_value("cyclegan")});
  options.add_option(group, {"test", "Will perform inference if set to true", value<bool>()->default_value("false")});

  options.add_option(group, {"dataset", "Path to dataset", value<std::string>()});

  // Model Options
  options.add_option(group, {"width", "Model input width", value<int>()->default_value("256")});
  options.add_option(group, {"height", "Model input height", value<int>()->default_value("256")});
  options.add_option(group, {"r,blocks", "Num resnet blocks", value<int>()->default_value("9")});

  // Training Options
  options.add_option(group, {"e,epochs", "Total epochs for training", value<int>()->default_value("500")});
  options.add_option(group, {"b,batch-size", "Batch size", value<int>()->default_value("4")});
  options.add_option(group, {"l,learning-rate", "Learning rate", value<double>()->default_value("0.0005")});
  options.add_option(group, {"lambda-identity", "Identity loss", value<double>()->default_value("0.5")});
  options.add_option(group, {"lambda-a", "Identity loss A", value<double>()->default_value("10.0")});
  options.add_option(group, {"lambda-b", "Identity loss b", value<double>()->default_value("10.0")});
  options.add_option(group, {"step-size", "Divide learning rate by", value<double>()->default_value("1.1")});

  options.add_option(group, {"latent-vector", "(nz) Size of the z latent vector", value<int>()->default_value("100")});

  // Export Options
  options.add_option(group, {"export-dir", "Export training results to", value<std::string>()->default_value("./")});

  auto result = options.parse(argc, argv);

  std::string trainer = result["trainer"].as<std::string>();
  if (trainer == "cyclegan"){ 
    CycleGAN::Trainer trainer(result);
    trainer.train();
  } else if (trainer == "dcgan") {
    DCGAN::Trainer trainer(result);
    trainer.train();
  } else {
    std::cout<<"Trainer "<<trainer<<" not implemented";
    exit(-1);
  }
  
  
}
