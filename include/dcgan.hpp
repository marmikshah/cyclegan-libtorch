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
      explicit ImageDataset(std::string directory, int width, int height) {
        for (const auto& entry : fs::directory_iterator(directory)) {
          imagePaths.push_back(entry.path().string());
        }
        std::cout << "Found " << imagePaths.size() << " images @ " << directory << std::endl;
        this->dims = cv::Size(height, width);
        std::cout << "Images will be resized to (h, w) => (" << height << ", " << width << ")" << std::endl;
      }

      Example get(size_t index) override {
        torch::Tensor sample = matToTensor(this->imagePaths[index], this->dims);

        return {sample, torch::scalar_tensor(1.0)};
      }

      torch::optional<size_t> size() const override { return this->imagePaths.size(); }
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
    TrainingOpts* opts;

   public:
    Trainer(cxxopts::ParseResult result) {
      opts = new TrainingOpts(result);
      dis = Models::createDiscriminator(64);
      gen = Models::createGenerator(opts->latentVector, 64);

      dis->to(device);
      gen->to(device);
      std::cout << " -------------------- Generator --------------------" << std::endl;
      std::cout << gen << std::endl;

      std::cout << " -------------------- Discriminator --------------------" << std::endl;
      std::cout << dis << std::endl;
    }

    void train() {
      using namespace torch::data;
      auto dataset = DataIO::ImageDataset(opts->datasetDir, opts->width, opts->height).map(transforms::Stack<>());
      auto loader = make_data_loader(std::move(dataset), DataLoaderOptions(opts->batchSize));

      namespace fs = std::filesystem;

      fs::path exportDir("./experiment/");
      std::string exportDirString = exportDir.string();
      fs::create_directory(exportDir);

      fs::path previewDir("./experiment/previews/");
      std::string previewDirString = previewDir.string();
      fs::create_directory(previewDir);

      using namespace torch::optim;
      std::cout << "Learning Rate: " << opts->learningRate << std::endl;
      Adam genOptimizer(gen->parameters(), AdamOptions(opts->learningRate));
      Adam disOptimizer(dis->parameters(), AdamOptions(opts->learningRate));
      std::cout << "------------------- Training Started -------------------" << std::endl;

      for (int64_t epoch = 1; epoch <= opts->maxEpochs; ++epoch) {
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

          torch::Tensor inputFake = torch::randn({batchSize, opts->latentVector, 1, 1}).to(device);
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
          torch::Tensor inputFake = torch::randn({1, opts->latentVector, 1, 1}).to(device);
          auto output = gen->forward(inputFake.detach());
          cv::imwrite(previewDirString + "generation" + strEpoch + ".png", tensorToMat(output[0], true));

          exportModel(gen, exportDirString + "/gencheckpoint" + strEpoch + ".pt");
        }

        std::cout << "Loss(G): " << epochGenLoss << ",\t";
        std::cout << "Loss(D): " << epochDisLoss << std::endl;
      }
    }
  };

};

#endif