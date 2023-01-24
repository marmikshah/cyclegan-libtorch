#ifndef ARTIUM_TRAIN_HPP
#define ARTIUM_TRAIN_HPP

#include <cxxopts.hpp>
#include <iostream>

#include "cyclegan.hpp"
#include "dataset.hpp"
#include "globals.hpp"
#include "imagetools.hpp"

namespace F = torch::nn::functional;

class Domain {
 public:
  std::shared_ptr<CycleGAN::Generator> generator;
  std::shared_ptr<CycleGAN::Discriminator> discriminator;
  torch::optim::Adam* optimizerG;
  torch::optim::Adam* optimizerD;
  ImagePool* pool;
  torch::Tensor fake, rec, real, idt;

  Domain(int numBlocks, double lr, int batchSize) {
    using namespace torch::nn;
    using namespace torch::optim;
    generator = std::make_shared<CycleGAN::Generator>(3, 3, 64, numBlocks);
    generator->to(device);
    discriminator = std::make_shared<CycleGAN::Discriminator>();
    discriminator->to(device);

    optimizerD = new Adam(discriminator->parameters(), AdamOptions().lr(lr).betas({0.5, 0.999}));
    optimizerG = new Adam(generator->parameters(), AdamOptions().lr(lr).betas({0.5, 0.999}));

    pool = new ImagePool(batchSize);
  }

  torch::Tensor generate(torch::Tensor& batch) { return generator->forward(batch); }

  torch::Tensor getFakeGenerations() { return pool->getImages(fake).to(device); }

  torch::Scalar trainDiscriminator(torch::Tensor& realBatch, torch::Tensor& fakeBatch) {
    setGrad(*discriminator, true);
    optimizerD->zero_grad();
    auto predReal = discriminator->forward(realBatch);
    auto predFake = discriminator->forward(fakeBatch);

    torch::Tensor lossReal = F::mse_loss(predReal, torch::ones_like(predReal));
    torch::Tensor lossFake = F::mse_loss(predFake, torch::zeros_like(predFake));

    auto loss = lossReal + lossFake * 0.5;
    loss.backward();
    optimizerD->step();

    return loss.item();
  }

  torch::Tensor computeGeneratorLoss(torch::Tensor& input, bool isReal = true) {
    torch::Tensor output = discriminator->forward(input);
    torch::Tensor target = isReal ? torch::ones_like(output) : torch::zeros_like(output);
    return F::mse_loss(output, target);
  }

  torch::Tensor computeCycleLoss(double lambda) { return F::mse_loss(rec, real) * lambda; }

  void exportGeneraton(std::string exportPath) {
    cv::Mat mat = tensorToMat(fake[0].detach().cpu().clone());
    cv::imwrite(exportPath, mat);
  }

  void cleanup(std::string exportDir, std::string identifier) {
    torch::serialize::OutputArchive archive;
    generator->save(archive);
    archive.save_to(exportDir + "/gen_" + identifier + ".pt");
  }
};

void train(cxxopts::ParseResult opts) {
  int numBlocks = opts["blocks"].as<int>();
  int learningRate = opts["learning-rate"].as<double>();
  int width = opts["width"].as<int>();
  int height = opts["height"].as<int>();
  int batchSize = opts["batch-size"].as<int>();
  std::string exportDir = opts["export-dir"].as<std::string>();

  Domain domainA(numBlocks, learningRate, batchSize);
  Domain domainB(numBlocks, learningRate, batchSize);

  using namespace torch::data;
  std::string directory = opts["dataset"].as<std::string>();
  ImageDataset dataset(directory + "/trainA", directory + "/trainB", width, height, batchSize);

  using namespace torch::nn;

  double lambdaIdt = opts["lambda-identity"].as<double>();
  double lambdaA = opts["lambda-a"].as<double>();
  double lambdaB = opts["lambda-b"].as<double>();

  std::cout << "------------------- Training Started -------------------" << std::endl;

  for (int epoch = 0; epoch < opts["epochs"].as<int>(); epoch++) {
    std::cout << "Epoch " << std::format("{:03}", epoch + 1) << ":\t";

    double epochGenLoss = 0.0, epochDALoss = 0.0, epochDBLoss = 0.0;
    int totalItemsA = 0, totalItemsB = 0;
    while (!dataset.isIterationComplete()) {
      Batch* batch = dataset.getBatch();

      domainA.real = batch->imagesA.to(device);
      domainB.real = batch->imagesB.to(device);
      totalItemsA += domainA.real.sizes()[0];
      totalItemsB += domainB.real.sizes()[0];

      domainB.fake = domainA.generate(domainA.real);  // G_A(A)
      domainA.rec = domainB.generate(domainB.fake);   // G_B(G_A(A))
      domainA.fake = domainB.generate(domainB.real);  // G_B(B)
      domainB.rec = domainA.generate(domainA.fake);   // G_A(G_B(B))

      domainA.idt = domainA.generate(domainB.real);
      domainB.idt = domainB.generate(domainA.real);

      setGrad(*domainA.discriminator, false);
      setGrad(*domainB.discriminator, false);
      domainA.optimizerG->zero_grad();
      domainB.optimizerG->zero_grad();

      torch::Tensor idtLossA = F::l1_loss(domainA.idt, domainB.real) * lambdaB * lambdaIdt;
      torch::Tensor idtLossB = F::l1_loss(domainB.idt, domainA.real) * lambdaA * lambdaIdt;
      torch::Tensor genLoss = domainA.computeGeneratorLoss(domainB.fake) + domainB.computeGeneratorLoss(domainA.fake);
      torch::Tensor cycleLoss = domainA.computeCycleLoss(lambdaA) + domainB.computeCycleLoss(lambdaB);

      torch::Tensor totalLoss = idtLossA + idtLossB + genLoss + cycleLoss;

      epochGenLoss += totalLoss.item().toDouble();
      totalLoss.backward();
      domainA.optimizerG->step();
      domainB.optimizerG->step();

      torch::Tensor fakeA = domainA.getFakeGenerations().detach();
      torch::Tensor fakeB = domainB.getFakeGenerations().detach();
      epochDALoss += domainA.trainDiscriminator(domainB.real, fakeB).toDouble();
      epochDBLoss += domainB.trainDiscriminator(domainA.real, fakeA).toDouble();  
    }
    std::cout << "Loss(G): " << epochGenLoss / (totalItemsA + totalItemsB) << ",\t";
    std::cout << "Loss(D_A): " << epochDALoss << ",\t";
    std::cout << "Loss(D_B): " << epochDBLoss << " \t";
    std::cout << std::endl;

    domainA.exportGeneraton(exportDir + "/GenA-" + std::to_string(epoch) + ".png");
    domainA.exportGeneraton(exportDir + "/GenA-" + std::to_string(epoch) + ".png");

    dataset.reset();
  }
  std::cout << "------------------- Training Complete -------------------" << std::endl;

  domainA.cleanup(exportDir, "A");
  domainB.cleanup(exportDir, "B");
}

#endif