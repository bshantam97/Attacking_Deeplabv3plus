from torch.nn.modules.batchnorm import BatchNorm2d
from .ASPP import *
from .resnet import *
import os 
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate = [12,24,36]):

        super(DeepLabHead, self).__init__()
        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256,256,3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])
    
    def _init_weight(self):
        # Returns an iterator over all the modules in the network
        for m in self.modules:
            # Returns if an object is an instance of a class or a subclass thereof
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,(nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate = [12,24,36]):

        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias = False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True)
        )

        self.aspp = ASPP(in_channels,aspp_dilate)

        # Here we have 304 input layers because the input from ASPP is 256*5 and using 1x1 conv it is reduced to 256 and then 
        # concatenated to the 48 channels of the low level features

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, features):
        low_level_features = self.project(features['low_level'])
        output_feature = self.aspp(features['out'])
        output_feature = F.interpolate(output_feature, size = low_level_features.shape[2:], mode = 'bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_features, ]))
    
    def _init_weight(self):
        # Returns an iterator over all the modules in the network
        for m in self.modules:
            # Returns if an object is an instance of a class or a subclass thereof
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,(nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeperableConvolution(nn.Module):
    
    " Atrous Seperable Convolution"
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, bias=True):
        super(AtrousSeperableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Seperable Convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride = stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            
            # Pointwise Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding=0, bias = bias)
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        # Returns an iterator over all the modules in the network
        for m in self.modules:
            # Returns if an object is an instance of a class or a subclass thereof
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,(nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def convert_to_seperable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeperableConvolution(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.bias
        )
    
    # Module is nothing but nn.Conv2d
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_seperable_conv(child))
    
    return new_module

class SimpleSegmentationModel(nn.Module):

    def __init__(self, backbone, classifier):
        super(SimpleSegmentationModel, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x):

        input_shape = x.shape[-2:]

        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)

        return x

class Deeplabv3(SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
    """

    pass

def segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride == 8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12,24,36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6,12,18]
    
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation
    )

    inplanes=2048
    low_level_planes=256

    if name == 'deeplabv3plus':
        return_layers = {'layer4':'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes , aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = Deeplabv3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone.startswith('resnet'):
        model = segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    
    else:
        raise NotImplementedError
    return model

def deeplabv3_resnet50(num_classes = 21, output_stride = 8, pretrained_backbone = True):

    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone = True):

    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):

    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):

    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)