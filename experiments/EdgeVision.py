import torch
import torch.nn as nn 
import timm
from collections import OrderedDict
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7

class EdgeVision(nn.Module):
    def __init__(self, num_classes=2, variant='mobilenetv4_conv_small', pretrained=True, img_size=224, fpn_channels=256):
        super().__init__()
        self.backbone = MobileNetV4(variant=variant, out_channels=fpn_channels, pretrained=pretrained)


    




class MobileNetV4(nn.Module):
    def __init__(self,variant='mobilenetv4_conv_small', out_channels=256, pretrained=True):
        super().__init__()

        self.body = timm.create_model(variant,features_only=True,out_indices=(2,3,4),pretrained=pretrained)

        in_channels_list = self.body.feature_info.channels()
        self.out_channels = out_channels

        
    def forward(self,x):
        feats = self.body(x)




class DetectionHead(nn.Module):
    def __init__(self,channel_in, B, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(channel_in,(5 + num_classes) * B, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class SSDHead(nn.Module):
    def __init__(self, out_channels, num_classes, anchors_list):
