import os
import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.utils import load_state_dict_from_url
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResidualBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample = None, stride = 1, dilation = 1):

        super(ResidualBasicBlock, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = dilation, dilation=dilation)
        self.BatchNorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = dilation, dilation = dilation)
        self.BatchNorm2 = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)

        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample = None, stride = 1, dilation = 1):

        super(ResidualBlock, self).__init__()
        # Expansion is used in the end of each residual block so that the number of channels increase by 4
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1, padding = 0)
        self.BatchNorm1 = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation)
        self.BatchNorm2 = nn.BatchNorm2d(num_features = out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.BatchNorm3 = nn.BatchNorm2d(num_features = out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.conv3(x)
        x = self.BatchNorm3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, image_channels, num_classes, replace_stride_with_dilation = None):

        super(ResNet, self).__init__()

        self.in_channels = 64 # The input to the residual blocks not the input to the model. 224x224x4 -> 112x112x64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation =  [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:

            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self.make_layer(block, layers[1], out_channels = 128, stride = 2, dilate = replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, layers[2], out_channels = 256, stride = 2, dilate = replace_stride_with_dilation[1])
        self.layer4 = self.make_layer(block, layers[3], out_channels = 512, stride = 2, dilate = replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512*4, num_classes)

    def make_layer(self, block, num_residual_blocks, out_channels, stride, dilate = False):
        
        identity_downsample = None
        layers = []
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels ,out_channels*4 ,kernel_size = 1, stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride, previous_dilation))
        self.in_channels = out_channels * 4 

        # Append the remaining layers
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels, dilation = self.dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

def resnet(architecture, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[architecture], progress = progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained = False, progress = True, **kwargs):

   return resnet('resnet18', ResidualBasicBlock, [2,2,2,2], pretrained, progress, **kwargs)

def resnet34(pretrained = False, progress = True, **kwargs):

    return resnet('resnet34', ResidualBasicBlock, [3,4,6,3], pretrained, progress, **kwargs)

def resnet50(pretrained = False, progress = True, **kwargs):

    return resnet('resnet50', ResidualBlock, [3,4,6,3], pretrained, progress, **kwargs)

def resnet101(pretrained = False, progress = True, **kwargs):

    return resnet('resnet101', ResidualBlock, [3,4,23,3], pretrained, progress, **kwargs)

def resnet152(pretrained = False, progress = True, **kwargs):

    return resnet('resnet152', ResidualBlock, [3,8,36,3], pretrained, progress, **kwargs)


# def test():

#     network = resnet(architecture = 'resnet50',block = ResidualBlock, layers = [3,4,6,3], pretrained = False, 
#                     progress = False, image_channels = 3, num_classes = 1000)
#     network = resnet50(image_channels = 3, num_classes = 1000, replace_stride_with_dilation = [False, True, False])
#     y = torch.randn(2,3,224,224)
#     y = network(y)
#     print(y.size())

# test()