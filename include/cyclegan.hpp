#ifndef ARTIUM_CYCLEGAN_HPP
#define ARTIUM_CYCLEGAN_HPP

#include <filesystem>

#include "blocks.hpp"
#include "globals.hpp"
#include "imagetools.hpp"
#include "structs.hpp"

namespace CycleGAN {

  using namespace torch::nn;

  struct Settings : SettingsBase {
    int numResnetBlocks;
    double lamndaIdentity;
    double lambdaA;
    double lambdaB;

    Settings(cxxopts::ParseResult opts) : SettingsBase(opts) {
      numResnetBlocks = opts["blocks"].as<int>();
      lamndaIdentity = opts["lambda-identity"].as<double>();
      lambdaA = opts["lambda-a"].as<double>();
      lambdaB = opts["lambda-b"].as<double>();
    }

    /* CycleGAN Specific Settings */
    int getPoolSize() { return 50; }

    std::string getDomainPath(std::string domain, std::string mode = "train") {
      return getDatasetDirectory() + "/" + mode + domain + "/";
    }
  };
  namespace Models {
    /**
     * ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
     * All models definitions have been taken from the repository mentioned above.
     */
    using namespace torch::nn;
    void initWeights(torch::nn::Module &module) {
      std::cout << module;
      torch::NoGradGuard noGrad;
      if (auto *layer = module.as<torch::nn::Conv2dImpl>()) {
        torch::nn::init::normal_(layer->weight, 0.0, 0.2);
      }

      if (auto *layer = module.as<torch::nn::InstanceNorm2dImpl>()) {
        torch::nn::init::normal_(layer->weight, 1.0, 0.2);
        torch::nn::init::constant_(layer->bias, 0.0);
      }
    }

    Sequential createDiscriminator(int inChannels = 3, int ndf = 64, int numLayers = 3) {
      /**
       * Initialize Discriminator Network
       *
       * @param inChannels Number of channels in the input images
       * @param ndf Number of filters in the last conv layer
       * @param numLayers Number of conv layers in the network
       */

      Sequential network;

      torch::ExpandingArray<2UL> kernel(4);
      int padding = 1;
      network->push_back(Conv2d(Conv2dOptions(inChannels, ndf, kernel).stride(2).padding(1)));
      network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      int multiplier = 1, multiplierPrev = 1;
      for (int i = 1; i < numLayers; i++) {
        multiplierPrev = multiplier;
        multiplier = min(1 << i, 8);
        network->push_back(Conv2d(Conv2dOptions(ndf * multiplierPrev, ndf * multiplier, kernel).stride(2).padding(1)));
        network->push_back(InstanceNorm2d(ndf * multiplier));
        network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));
      }

      multiplierPrev = multiplier;
      multiplier = min(1 << numLayers, 8);
      network->push_back(Conv2d(Conv2dOptions(ndf * multiplierPrev, ndf * multiplier, kernel).stride(1).padding(1)));
      network->push_back(InstanceNorm2d(ndf * multiplier));
      network->push_back(LeakyReLU(LeakyReLUOptions().negative_slope(0.2).inplace(true)));

      network->push_back(Conv2d(Conv2dOptions(ndf * multiplier, 1, kernel).stride(1).padding(padding)));

      return network;
    }

    Sequential createGenerator(int inChannels, int outChannels, int ngf = 64, int numBlocks = 6) {
      /**
       * Initialise a Generater Network.
       * @param inChannels Number of channels of input image (3 for Color Image and 1 for Grayscale)
       * @param outChannles Number of channels of outpit image (3 for Color Image and 1 for Grayscale)
       * @param nfg Number of filters in the last convolution layer
       * @param numBlocks Number of Residual blocks in the network.
       *
       * NOTE: Bias is disabled
       */
      Sequential network;
      /* ========== Encoder ========== */

      network->push_back(ReflectionPad2d(3));
      network->push_back(Conv2d(Conv2dOptions(inChannels, ngf, 7).padding(0).stride(1)));
      network->push_back(InstanceNorm2d(ngf));
      network->push_back(ReLU(true));

      int totalDownsamplingLayers = 2;
      int multiplier = 0;
      for (int i = 0; i < totalDownsamplingLayers; i++) {
        multiplier = 1 << i;
        int inFeatures = ngf * multiplier, outFeatures = ngf * multiplier * 2;
        network->push_back(Conv2d(Conv2dOptions(inFeatures, outFeatures, 3).stride(2).padding(1)));
        network->push_back(InstanceNorm2d(outFeatures));
        network->push_back(ReLU(true));
      }

      /* ========== Transformer ========== */
      multiplier = 1 << totalDownsamplingLayers;
      for (int i = 0; i < numBlocks; i++) {
        ResidualBlock block(ngf * multiplier);
        network->push_back(block);
      }

      /* ========== Decoder ========== */
      for (int i = 0; i < totalDownsamplingLayers; i++) {
        multiplier = 1 << (totalDownsamplingLayers - i);
        int inFeatures = ngf * multiplier, outFeatures = ngf * multiplier / 2;
        network->push_back(
            ConvTranspose2d(ConvTranspose2dOptions(inFeatures, outFeatures, 3).stride(2).padding(1).output_padding(1)));
        network->push_back(InstanceNorm2d(outFeatures));
        network->push_back(ReLU(true));
      }

      network->push_back(ReflectionPad2d(3));
      network->push_back(Conv2d(Conv2dOptions(ngf, outChannels, 7).padding(0)));
      network->push_back(Tanh());
      // network->apply(initWeights);
      return network;
    }

  };

  namespace DataIO {
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
      explicit ImageDataset(Settings opts) {
        for (const auto &entry : fs::directory_iterator(opts.getDomainPath("A"))) {
          if (entry.path().string().size() == 0) continue;
          pathsA.push_back(entry.path().string());
        }
        std::cout << "Found " << pathsA.size() << " samples for Domain A" << std::endl;
        for (const auto &entry : fs::directory_iterator(opts.getDomainPath("B"))) {
          if (entry.path().string().size() == 0) continue;
          pathsB.push_back(entry.path().string());
        }
        std::cout << "Found " << pathsB.size() << " samples for Domain B" << std::endl;
        this->dims = opts.getInputSize();
      }

      Example get(size_t index) override {
        /**
         * As C++ Frontend does not provide a way to return Map,
         * we use `target` of the Example<> as the batch for
         * Domain B.
         *
         * @param index Index of image paths to use.
         */

        torch::Tensor sampleA = matToTensor(this->pathsA[index % pathsA.size()], this->dims);
        torch::Tensor sampleB = matToTensor(this->pathsB[index % pathsB.size()], this->dims);

        return {sampleA, sampleB};
      }

      torch::optional<size_t> size() const override { return min(this->pathsA.size(), this->pathsB.size()); }
    };
  };

  class Trainer {
   private:
    Sequential genA, genB, disA, disB;
    torch::optim::Adam *optimGA, *optimGB, *optimDA, *optimDB;
    Settings opts;
    ImagePool *poolA, *poolB;

   public:
    Trainer(cxxopts::ParseResult result) : opts(result) {
      genA = Models::createGenerator(3, 3, 64, opts.numResnetBlocks);
      genB = Models::createGenerator(3, 3, 64, opts.numResnetBlocks);
      disA = Models::createDiscriminator();
      disB = Models::createDiscriminator();

      genA->to(device);
      genB->to(device);
      disA->to(device);
      disB->to(device);

      using namespace torch::optim;
      double lr = opts.getLearningRate();
      std::cout << "Optimizer Learning Rate: " << lr << std::endl;
      optimGA = new Adam(genA->parameters(), AdamOptions(lr).betas({0.5, 0.999}));
      optimDA = new Adam(disA->parameters(), AdamOptions(lr).betas({0.5, 0.999}));
      optimGB = new Adam(genB->parameters(), AdamOptions(lr).betas({0.5, 0.999}));
      optimDB = new Adam(disB->parameters(), AdamOptions(lr).betas({0.5, 0.999}));

      poolA = new ImagePool(50);
      poolB = new ImagePool(50);
    }

    double updateLR(torch::optim::Adam &optim) {
      double updatedLR = -1;
      for (auto &group : optim.param_groups()) {
        if (group.has_options()) {
          auto &options = static_cast<torch::optim::OptimizerOptions &>(group.options());
          auto lr = options.get_lr();
          options.set_lr(lr * opts.getStepGamma());
          if (updatedLR < 0) {
            updatedLR = options.get_lr();
          }
        }
      }
      return updatedLR;
    }

    void train() {
      using namespace torch::data;
      auto dataset = DataIO::ImageDataset(opts).map(transforms::Stack<>());
      auto loader = make_data_loader(std::move(dataset), DataLoaderOptions(opts.getBatchSize()));

      namespace fs = std::filesystem;

      std::string exportDir = opts.getExperimentDirectory();
      std::string previewDir = opts.getPreviewsDirectory();

      int totalIterations = 0;
      std::cout << "------------------- Training Started -------------------" << std::endl;
      for (int epoch = 1; epoch <= opts.getTotalEpochs(); epoch++) {
        std::cout << "Epoch " << epoch << ":\t";
        double epochGenLoss = 0.0, epochDLoss = 0.0;
        std::string strEpoch = std::to_string(epoch);
        for (torch::data::Example<> &batch : *loader) {
          totalIterations += 1;
          torch::Tensor realImagesA = batch.data.to(device);
          torch::Tensor realImagesB = batch.target.to(device);

          /* =========================== Train Generators =========================== */
          genA->zero_grad();
          genB->zero_grad();
          setGrad(disA, false);
          setGrad(disB, false);

          // G_A(A) => G_B(G_A(A))
          torch::Tensor fakeImagesB = genA->forward(realImagesA);
          torch::Tensor reconstructedImagesA = genB->forward(fakeImagesB);

          // G_B(B) => G_A(G_B(B))
          torch::Tensor fakeImagesA = genB->forward(realImagesB);
          torch::Tensor reconstructedImagesB = genA->forward(fakeImagesA);

          torch::Tensor identityA = genA->forward(realImagesB);
          torch::Tensor identityB = genB->forward(realImagesA);

          torch::Tensor identityLossA =
              functional::l1_loss(identityA, realImagesA) * opts.lambdaA * opts.lamndaIdentity;
          torch::Tensor identityLossB =
              functional::l1_loss(identityB, realImagesB) * opts.lambdaB * opts.lamndaIdentity;

          // Forward pass on D
          torch::Tensor disAOnFakeB = disA->forward(fakeImagesB);
          torch::Tensor disBOnFakeA = disB->forward(fakeImagesA);

          torch::Tensor _targetReal = torch::ones_like(disAOnFakeB).to(device);

          torch::Tensor generatorLoss =
              functional::mse_loss(disAOnFakeB, _targetReal) + functional::mse_loss(disAOnFakeB, _targetReal);

          torch::Tensor cycleLossA = functional::l1_loss(reconstructedImagesA, realImagesA) * opts.lambdaA;
          torch::Tensor cycleLossB = functional::l1_loss(reconstructedImagesB, realImagesB) * opts.lambdaB;

          torch::Tensor totalGenLoss = generatorLoss + cycleLossA + cycleLossB + identityLossA + identityLossB;
          assert(totalGenLoss.requires_grad());

          totalGenLoss.backward();
          optimGA->step();
          optimGB->step();

          epochGenLoss += (totalGenLoss.item().toDouble() / (realImagesA.size(0) + realImagesB.size(0)));

          /* ======================  Train Discriminators ====================== */

          setGrad(disA, true);
          setGrad(disB, true);
          disA->zero_grad();
          disB->zero_grad();

          torch::Tensor _fakeImagesB = poolB->getImages(fakeImagesB);
          torch::Tensor disAPredRealB = disA->forward(realImagesB);
          torch::Tensor disLossA = functional::mse_loss(disAPredRealB, torch::ones_like(disAPredRealB).to(device));
          torch::Tensor disAPredFakeB = disA->forward(_fakeImagesB.detach());
          disLossA += functional::mse_loss(disAPredFakeB, torch::zeros_like(disAPredFakeB).to(device));
          disLossA *= 0.5;
          disLossA.backward();

          torch::Tensor _fakeImagesA = poolA->getImages(fakeImagesA);
          torch::Tensor disBPredRealA = disB->forward(realImagesA);
          torch::Tensor disLossB = functional::mse_loss(disBPredRealA, torch::ones_like(disBPredRealA).to(device));
          torch::Tensor disBPredFakeA = disB->forward(_fakeImagesA.detach());
          disLossB += functional::mse_loss(disBPredFakeA, torch::ones_like(disBPredFakeA).to(device));
          disLossB *= 0.5;
          disLossB.backward();

          optimDA->step();
          optimDB->step();

          epochDLoss +=
              ((disLossA.item().toDouble() / fakeImagesB.size(0)) + (disLossB.item().toDouble() / fakeImagesA.size(0)));

          cv::imwrite(previewDir + "/realA-" + strEpoch + ".png", tensorToMat(realImagesA[0], true));
          cv::imwrite(previewDir + "/realB-" + strEpoch + ".png", tensorToMat(realImagesB[0], true));
          cv::imwrite(previewDir + "/fakeA-" + strEpoch + ".png", tensorToMat(fakeImagesA[0], true));
          cv::imwrite(previewDir + "/fakeB-" + strEpoch + ".png", tensorToMat(fakeImagesB[0], true));

          if (totalIterations % opts.getStepSize() == 0) {
            std::cout << "Updated LR:" << updateLR(*optimGA) << "\t";
            updateLR(*optimGB);
            updateLR(*optimDA);
            updateLR(*optimDB);
          }
        }

        std::cout << "Loss(G): " << epochGenLoss << ",\t";
        std::cout << "Loss(D): " << epochDLoss << std::endl;

        exportModel(genA, exportDir + "/genAcheckpoint" + strEpoch + ".pt");
        exportModel(genB, exportDir + "/genBcheckpoint" + strEpoch + ".pt");
      }
    }
  };

};

#endif