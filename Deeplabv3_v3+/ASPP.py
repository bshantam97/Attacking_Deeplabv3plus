import os
import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = dilation, dilation = dilation),
                   nn.BatchNorm2d(num_features = out_channels),
                   nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)

        """
        Basically applies global average pooling to the last feature map, passes it through a 1x1 convolution with 256 filters
        and batch normalization and then bilinearly upsample the image to the desired spatial dimension. 
        """
class ASPPPool(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPool, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1),
                                        nn.BatchNorm2d(num_features = out_channels), nn.ReLU())
    
    def forward(self,x):
        size = x.shape[-2:] # For upsampling
        x = super(ASPPPool, self).forward(x)
        return torch.nn.functional.interpolate(x, size, mode = 'bilinear', align_corners = False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels = 256):
        super(ASPP, self).__init__()
        modules = []

        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                       nn.BatchNorm2d(out_channels), nn.ReLU()))
        
        atrous_rate1, atrous_rate2, atrous_rate3 = tuple(atrous_rates)

        modules.append(ASPPConv(in_channels, out_channels, atrous_rate1))
        modules.append(ASPPConv(in_channels, out_channels, atrous_rate2))
        modules.append(ASPPConv(in_channels, out_channels, atrous_rate3))
        modules.append(ASPPPool(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Input channels is 5*in_channels because in the ASPP layer thr output feature maps are being concatenated
        self.project = nn.Sequential(
            nn.Conv2d(5*in_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self,x):

        aspp_out= []

        for conv in self.convs:
            aspp_out.append(conv(x))
        aspp_out = torch.cat(aspp_out, dim = 1)

        return self.project(aspp_out)

def test():

    aspp = ASPP(in_channels = 256, atrous_rates = [6,12,18])
    y = torch.randn(2,256,14,14)
    y = aspp(y)
    print(y.size())

test()