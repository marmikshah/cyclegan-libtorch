#ifndef ARTIUM_TRAIN_HPP
#define ARTIUM_TRAIN_HPP

#include <cxxopts.hpp>
#include <iostream>

#include "cyclegan.hpp"
#include "dataset.hpp"
#include "globals.hpp"
#include "imagetools.hpp"

class Domain {
 public:
  std::shared_ptr<CycleGAN::Generator> generator;
  std::shared_ptr<CycleGAN::Discriminator> discriminator;
  torch::optim::Adam* optimizerG;
  torch::optim::Adam* optimizerD;
  ImagePool* pool;
  std::vector<cv::Mat> generations;
  torch::Tensor fake, rec, real, idt;

  Domain(int numBlocks, double lr) {
    using namespace torch::nn;
    using namespace torch::optim;
    generator = std::make_shared<CycleGAN::Generator>(3, 3, 64, numBlocks);
    generator->to(device);
    discriminator = std::make_shared<CycleGAN::Discriminator>();
    discriminator->to(device);

    optimizerD = new Adam(discriminator->parameters(), AdamOptions().lr(lr).betas({0.5, 0.999}));
    optimizerG = new Adam(generator->parameters(), AdamOptions().lr(lr).betas({0.5, 0.999}));

    pool = new ImagePool(50);
  }

  torch::Tensor generate(torch::Tensor& batch) { return generator->forward(batch); }

  torch::Tensor getFakeGenerations() { return pool->getImages(fake).to(device); }

  torch::Scalar trainDiscriminator(torch::Tensor realBatch, torch::Tensor fakeBatch) {
    setGrad(*discriminator, true);
    optimizerD->zero_grad();
    auto predReal = discriminator->forward(realBatch);
    auto predFake = discriminator->forward(fakeBatch);

    using namespace torch::nn::functional;
    torch::Tensor lossReal = torch::nn::functional::mse_loss(predReal, torch::ones_like(predReal));
    torch::Tensor lossFake = torch::nn::functional::mse_loss(predFake, torch::zeros_like(predFake));

    auto loss = lossReal + lossFake * 0.5;
    loss.backward();
    optimizerD->step();

    return loss.item();
  }

  void step() { generations.push_back(tensorToMat(fake.detach().cpu()[0].clone())); }

  void cleanup(std::string exportDir, std::string identifier) {
    torch::serialize::OutputArchive archive;
    generator->save(archive);
    archive.save_to(exportDir + "/gen_" + identifier + ".pt");

    std::string frameExportPath = exportDir + "/" + identifier;
    std::cout << "Exporting " << generations.size() << " generations" << std::endl;

    for (int i = 0; i < generations.size(); i++) {
      auto& mat = generations[i];
      std::string path = frameExportPath + std::to_string(i) + ".png";
      cv::imwrite(path, mat);
    }
  }
};

void train(cxxopts::ParseResult opts) {
  int numBlocks = opts["blocks"].as<int>();
  int learningRate = opts["learning-rate"].as<double>();

  Domain domainA(numBlocks, learningRate);
  Domain domainB(numBlocks, learningRate);

  using namespace torch::data;
  datasets::MapDataset dataset = ImageDataset(opts["dataset"].as<std::string>()).map(transforms::Stack<>());
  std::unique_ptr loader = make_data_loader(std::move(dataset), 2);

  using namespace torch::nn;

  double lambdaIdt = opts["lambda-identity"].as<double>();
  double lambdaA = opts["lambda-a"].as<double>();
  double lambdaB = opts["lambda-b"].as<double>();

  std::cout << "------------------- Training Started -------------------" << std::endl;

  for (int epoch = 0; epoch < opts["epochs"].as<int>(); epoch++) {
    std::cout << "Epoch " << epoch << ":\t";
    for (auto& batch : *loader) {
      domainA.real = batch.data[0].unsqueeze(0);
      domainB.real = batch.data[1].unsqueeze(0);

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

      torch::Tensor targetReal = torch::ones({1, 1, 16, 16}).to(device);

      torch::Tensor idtLossA = functional::l1_loss(domainA.idt, domainB.real) * lambdaB * lambdaIdt;
      torch::Tensor idtLossB = functional::l1_loss(domainB.idt, domainA.real) * lambdaA * lambdaIdt;
      torch::Tensor genLoss = functional::mse_loss(domainA.discriminator->forward(domainB.fake), targetReal) +
                              functional::mse_loss(domainB.discriminator->forward(domainA.fake), targetReal) +
                              (functional::l1_loss(domainA.fake, batch.data) * lambdaA) +
                              (functional::l1_loss(domainB.fake, batch.data) * lambdaB) + idtLossA + idtLossB;
      std::cout << "Loss(G): " << genLoss.item() << ",\t";
      genLoss.backward();
      domainA.optimizerG->step();
      domainB.optimizerG->step();

      torch::Tensor fakeA = domainA.getFakeGenerations().detach();
      torch::Tensor fakeB = domainB.getFakeGenerations().detach();
      std::cout << "Loss(D_A): " << domainA.trainDiscriminator(domainB.real, fakeB) << ",\t";
      std::cout << "Loss(D_B): " << domainB.trainDiscriminator(domainA.real, fakeA) << ",\t";
      std::cout << std::endl;
      domainA.step();
      domainB.step();
    }
  }
  std::cout << "------------------- Training Complete -------------------" << std::endl;
  std::string exportDir = opts["export-dir"].as<std::string>();

  domainA.cleanup(exportDir, "A");
  domainB.cleanup(exportDir, "B");
}

#endif