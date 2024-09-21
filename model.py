import torch
from torchvision import models
import torchvision.models.segmentation as segmentation
import numpy as np
def Unet(output_channels, in_channels):
    model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    # Adjust the input and output layers accordingly
    model.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = torch.nn.Conv2d(256, output_channels, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, output_channels, kernel_size=1)
    return model