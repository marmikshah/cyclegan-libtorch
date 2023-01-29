#ifndef ARTIUM_DCGAN_HPP
#define ARTIUM_DCGAN_HPP

#include <cxxopts.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "globals.hpp"
#include "imagetools.hpp"
#include "structs.hpp"

namespace DCGAN {

  using namespace torch::nn;

  struct Settings : SettingsBase {
    int nz;
    Settings(cxxopts::ParseResult opts) : SettingsBase(opts) { nz = opts["latent-vector"].as<int>(); }
  };

  namespace DataIO {
    namespace fs = std::filesystem;

    struct ImageDataset : public torch::data::datasets::Dataset<ImageDataset> {
      /**
       * Custom dataset class for DCGAN
       */
      using Example = torch::data::Example<>;
      std::vector<std::string> imagePaths;
      cv::Size dims;

     public:
      explicit ImageDataset(Settings& opts) {
        std::string directory = opts.getDatasetDirectory();
        for (const auto& entry : fs::directory_iterator(directory)) {
          imagePaths.push_back(entry.path().string());
        }
        std::cout << "Found " << imagePaths.size() << " images @ " << directory << std::endl;
        dims = opts.getInputSize();
        std::cout << "Images will be resized to (h, w) => (" << dims << ")" << std::endl;
      }

      Example get(size_t index) override {
        torch::Tensor sample = matToTensor(imagePaths[index], dims);

        return {sample, torch::scalar_tensor(1.0)};
      }

      torch::optional<size_t> size() const override { return imagePaths.size(); }
    };
  };

  namespace Models {
    using namespace torch::nn;

    Sequential createGenerator(int nz, int ngf) {
      Sequential network;

      network->push_back(ConvTranspose2d(ConvTranspose2dOptions(nz, ngf * 8, 4).stride(1).padding(0).bias(false)));
      network->push_back(BatchNorm2d(ngf * 8));
      network->push_back(ReLU(ReLUOptions().inplace(true)));

      network->push_back(ConvTranspose2d(ConvTranspose2dOptions(ngf * 8, ngf * 4, 4).stride(2).padding(1).bias(false)));
      network->push_back(BatchNorm2d(ngf * 4));
      network->push_back(ReLU(ReLUOptions().inplace(true)));

      network->push_back(ConvTranspose2d(ConvTranspose2dOptions(ngf * 4, ngf * 2, 4).stride(2).padding(1).bias(false)));
      network->push_back(BatchNorm2d(ngf * 2));
      network->push_back(ReLU(ReLUOptions().inplace(true)));

      network->push_back(ConvTranspose2d(ConvTranspose2dOptions(ngf * 2, ngf, 4).stride(2).padding(1).bias(false)));
      network->push_back(BatchNorm2d(ngf));
      network->push_back(ReLU(ReLUOptions().inplace(true)));

      network->push_back(ConvTranspose2d(ConvTranspose2dOptions(ngf, 3, 4).stride(2).padding(1).bias(false)));
      network->push_back(Tanh());
      return network;
    }

    Sequential createDiscriminator(int ndf) {
      Sequential network;
      network->push_back(Conv2d(Conv2dOptions(3, ndf, 4).stride(2).padding(1).bias(false)));
      network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      network->push_back(Conv2d(Conv2dOptions(ndf, ndf * 2, 4).stride(2).padding(1).bias(false)));
      network->push_back(BatchNorm2d(ndf * 2));
      network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      network->push_back(Conv2d(Conv2dOptions(ndf * 2, ndf * 4, 4).stride(2).padding(1).bias(false)));
      network->push_back(BatchNorm2d(ndf * 4));
      network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      network->push_back(Conv2d(Conv2dOptions(ndf * 4, ndf * 8, 4).stride(2).padding(1).bias(false)));
      network->push_back(BatchNorm2d(ndf * 8));
      network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      network->push_back(Conv2d(Conv2dOptions(ndf * 8, 1, 4).stride(1).padding(0).bias(false)));
      network->push_back(Sigmoid());

      return network;
    }

  };

  class Trainer {
    Sequential gen, dis;
    Settings opts;

   public:
    Trainer(cxxopts::ParseResult result) : opts(result) {
      dis = Models::createDiscriminator(64);
      gen = Models::createGenerator(opts.nz, 64);

      dis->to(device);
      gen->to(device);
      std::cout << " -------------------- Generator --------------------" << std::endl;
      std::cout << gen << std::endl;

      std::cout << " -------------------- Discriminator --------------------" << std::endl;
      std::cout << dis << std::endl;
    }

    void train() {
      using namespace torch::data;
      auto dataset = DataIO::ImageDataset(opts).map(transforms::Stack<>());
      auto loader = make_data_loader(std::move(dataset), DataLoaderOptions(opts.getBatchSize()));

      namespace fs = std::filesystem;

      std::string exportDir = opts.getExperimentDirectory();
      std::string previewDir = opts.getPreviewsDirectory();

      using namespace torch::optim;
      double lr = opts.getLearningRate();

      std::cout << "Learning Rate: " << lr << std::endl;
      Adam genOptimizer(gen->parameters(), AdamOptions(lr));
      Adam disOptimizer(dis->parameters(), AdamOptions(lr));
      std::cout << "------------------- Training Started -------------------" << std::endl;

      for (int64_t epoch = 1; epoch <= opts.getTotalEpochs(); ++epoch) {
        double epochGenLoss = 0.0, epochDisLoss = 0.0;
        std::string strEpoch = std::to_string(epoch);
        std::cout << "Epoch " << epoch << ":\t";

        for (torch::data::Example<>& batch : *loader) {
          dis->zero_grad();

          torch::Tensor inputReal = batch.data.to(device);
          int batchSize = inputReal.size(0);
          torch::Tensor predReal = dis->forward(inputReal);
          torch::Tensor target = torch::full_like(predReal, 1.0).to(device);
          torch::Tensor disLossReal = torch::binary_cross_entropy(predReal, target);
          disLossReal.backward();

          torch::Tensor inputFake = torch::randn({batchSize, opts.nz, 1, 1}).to(device);
          torch::Tensor fakeGenerations = gen->forward(inputFake);
          target.fill_(0.0);
          torch::Tensor predFake = dis->forward(fakeGenerations.detach());
          torch::Tensor disLossFake = torch::binary_cross_entropy(predFake, target);
          disLossFake.backward();

          torch::Tensor totalDisLoss = disLossFake + disLossReal;
          epochDisLoss += totalDisLoss.item().toDouble();
          disOptimizer.step();

          gen->zero_grad();
          target.fill_(1);
          predFake = dis->forward(fakeGenerations);
          torch::Tensor genLoss = torch::binary_cross_entropy(predFake, target);
          genLoss.backward();
          genOptimizer.step();

          epochGenLoss += genLoss.item().toDouble();
        }

        if (epoch % 5 == 0) {
          torch::Tensor inputFake = torch::randn({1, opts.nz, 1, 1}).to(device);
          auto output = gen->forward(inputFake.detach());
          cv::imwrite(previewDir + "generation" + strEpoch + ".png", tensorToMat(output[0], true));

          exportModel(gen, exportDir + "/gencheckpoint" + strEpoch + ".pt");
        }

        std::cout << "Loss(G): " << epochGenLoss << ",\t";
        std::cout << "Loss(D): " << epochDisLoss << std::endl;
      }
    }
  };

};

#endif