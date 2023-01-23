#ifndef ARTIUM_UTILS_HPP
#define ARTIUM_UTILS_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


inline torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

inline int min(int a, int b) { return a < b ? a : b; }

void setGrad(torch::nn::Module& model, bool grad) {
    for(auto& param: model.parameters()){
        param.set_requires_grad(grad);
    }
}

bool saveImage(torch::Tensor tensor, std::string imagePath) {
    tensor = tensor.detach().permute({1, 2, 0}).contiguous();
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8).to(torch::kCPU);
    cv::Mat output(256, 256, CV_8UC3, tensor.data_ptr<uchar>());
    return cv::imwrite(imagePath, output);
}

#endif