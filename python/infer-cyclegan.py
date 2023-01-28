import os
import argparse

import cv2
import torch
from torch import nn

import numpy as np

def normalize(tensor):
    mean = torch.ones((3, 256, 256)) * 0.5
    std = torch.ones((3, 256, 256)) * 0.5
    tensor = tensor / 255.0
    tensor = (tensor - mean) / std
    return tensor

def denormalize(tensor):
    return tensor.add(1).div_(2).clamp_(0, 1).mul(255).add(0.5).clamp(0, 255)

class ResidualBlock(torch.nn.Module):
    def __init__(self, features):
        """
            Resnet block used as transformation layers in the
            Generator.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features)
            )

    def forward(self, target):
        return target + self.conv(target)

def create_generator(inChannels, outChannels, ngf=64, num_blocks=9):
    layers = []

    layers.append(nn.ReflectionPad2d(2))
    layers.append(nn.Conv2d(inChannels, ngf, 7, padding=0, stride=1))
    layers.append(nn.InstanceNorm2d(ngf))
    layers.append(nn.ReLU(True))

    totalDownsamplingLayers = 2
    multiplier = 0
    for i in range(totalDownsamplingLayers):
        multiplier = 1 << i
        inFeatures = ngf * multiplier
        outFeatures = ngf * multiplier * 2
        layers.append(nn.Conv2d(inFeatures, outFeatures, 3, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(outFeatures))
        layers.append(nn.ReLU(True))
    

    multiplier = 1 << totalDownsamplingLayers
    for i in range(num_blocks):
        layers.append(ResidualBlock(ngf * multiplier))

    for i in range(totalDownsamplingLayers):
        multiplier = 1 << (totalDownsamplingLayers - i)
        inFeatures = ngf * multiplier
        outFeatures = int((ngf * multiplier) / 2)
        layers.append(nn.ConvTranspose2d(inFeatures, outFeatures, 3, stride=2, padding=1, output_padding=1))
        layers.append(nn.InstanceNorm2d(outFeatures))
        layers.append(nn.ReLU(True))

    layers.append(nn.ReflectionPad2d(3))
    layers.append(nn.Conv2d(ngf, outChannels, 7, padding=0))
    layers.append(nn.Tanh())

    return nn.Sequential(*layers)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--input-image', required=True)
    parser.add_argument('-d', '--result-path', default="result.png")
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-r', '--num-block', type=int, default=9)


    args = parser.parse_args()

    assert os.path.exists(args.input_image)
    assert os.path.exists(args.checkpoint)

    net = create_generator(3, 3)
    net.load_state_dict(torch.jit.load(args.checkpoint).state_dict())

    net = net.cuda()

    mat = cv2.imread(args.input_image)
    mat = cv2.resize(mat, (256, 256))
    
    real = normalize(torch.Tensor(mat).permute(2, 1, 0)).unsqueeze(0).cuda()
    generated = denormalize(net(real).detach().cpu()[0]).permute(2, 1, 0)
    generated = np.array(generated, dtype=np.uint8)
    
    cv2.imwrite(args.result_path, generated)
    
