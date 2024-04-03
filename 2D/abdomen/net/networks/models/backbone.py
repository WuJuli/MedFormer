import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import logging

from scipy import ndimage


from .MaxViT import maxvit_tiny_rw_224, maxvit_rmlp_small_rw_224

class MaxViT_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(MaxViT_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if pretrain:
            self.backbone = load_pretrained_weights('maxxvit', model_scale)
        else:
            if model_scale == 'tiny':
                self.backbone = maxvit_tiny_rw_224()
            elif model_scale == 'small':
                self.backbone = maxvit_rmlp_small_rw_224()
            else:
                sys.exit(model_scale + " is not a valid model scale !")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]

from .Biformer import biformer_tiny, biformer_small, biformer_base

class Biformer_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(Biformer_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if model_scale == 'tiny':
            self.backbone = biformer_tiny(pretrained=pretrain)
        elif model_scale == 'small':
            self.backbone = biformer_small(pretrained=pretrain)
        elif model_scale == 'base':
            self.backbone = biformer_base(pretrained=pretrain)
        else:
            sys.exit(model_scale + " is not a valid model scale !")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]
    
from .ConvNeXt import convnext_tiny, convnext_small

class ConvNeXt_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(ConvNeXt_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if pretrain:
            self.backbone = load_pretrained_weights('convnext', model_scale)
        else:
            if model_scale == 'tiny':
                self.backbone = convnext_tiny()
            elif model_scale == 'small':
                self.backbone = convnext_small()
            else:
                sys.exit(model_scale + " is not a valid model scale !")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]
    

from .HorNet import hornet_tiny_7x7, hornet_tiny_gf, hornet_small_7x7, hornet_small_gf

class HorNet_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(HorNet_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if pretrain:
            self.backbone = load_pretrained_weights('hornet', model_scale)
        else:
            if model_scale == 'tiny-7':
                self.backbone = hornet_tiny_7x7()
            elif model_scale == 'tiny-gf':
                self.backbone = hornet_tiny_gf()
            elif model_scale == 'small-7':
                self.backbone = hornet_small_7x7()
            elif model_scale == 'small-gf':
                self.backbone = hornet_small_gf()
            else:
                sys.exit(model_scale + " is not a valid model scale ! ")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]
    

from .InceptionNext import inception_next_tiny, inception_next_small

class InceptionNext_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(InceptionNext_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if pretrain:
            self.backbone = load_pretrained_weights('inceptionnext', model_scale)
        else:
            if model_scale == 'tiny':
                self.backbone = inception_next_tiny()
            elif model_scale == 'small':
                self.backbone = inception_next_small()
            else:
                 sys.exit(model_scale + " is not a valid model scale !")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]
    
from .RepViT import repvit_m1_1, repvit_m1_5

class RepViT_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(RepViT_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if pretrain:
            self.backbone = load_pretrained_weights('repvit', model_scale)
        else:
            if model_scale == 'm11-300e' or 'm11-450e':
                self.backbone = repvit_m1_1()
            elif model_scale == 'm15-300e' or 'm15-45e':
                self.backbone = repvit_m1_5()
            else:
                 sys.exit(model_scale + " is not a valid model scale !")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]
    

from .SwinTransformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224

class SwinTransformer_Out(nn.Module):
    def __init__(self, model_scale, pretrain):
        super(SwinTransformer_Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if pretrain:
            self.backbone = load_pretrained_weights('swintransformer', model_scale)
        else:
            if model_scale == 'tiny':
                self.backbone = swin_tiny_patch4_window7_224()
            elif model_scale == 'small':
                self.backbone = swin_small_patch4_window7_224()
            else:
                 sys.exit(model_scale + " is not a valid model scale !")

    def forward(self, x):

        if x.size()[1] == 1:
            x = self.conv(x)

        f = self.backbone(x)

        return f[3], f[2], f[1], f[0]
    
def load_pretrained_weights(model_type, model_scale):
    if model_type == 'maxxvit':
        if model_scale == 'tiny':
            backbone = maxvit_tiny_rw_224()
            print('Loading:', 'Maxvit tiny')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
        elif model_scale == 'small':
            backbone = maxvit_rmlp_small_rw_224()
            print('Loading:', 'Maxvit small')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        else:
            sys.exit(model_scale + " is not a valid model scale! Maxxvit currently supported model scales are 'tiny' and 'small'.")
        
    if model_type == 'convnext':
        if model_scale == 'tiny':
            backbone = convnext_tiny()
            print('Loading:', 'convnext tiny')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/ConvNeXt/convnext_tiny_1k_224.pth')
        elif model_scale == 'small':
            backbone = convnext_small()
            print('Loading:', 'convnext small')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/ConvNeXt/convnext_small_1k_224.pth')
        else:
            sys.exit(model_scale + " is not a valid model scale! Convnext currently supported model scales are 'tiny' and 'small'.")
    
    if model_type == 'hornet':
        if model_scale == 'tiny-7':
            backbone = hornet_tiny_7x7()
            print('Loading:', 'hornet tiny-7')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/HorNet/hornet_tiny_7x7.pth')
        elif model_scale == 'tiny-gf':
            backbone = hornet_tiny_gf()
            print('Loading:', 'hornet tiny-gf')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/HorNet/hornet_tiny_gf.pth')
        elif model_scale == 'small-7':
            backbone = hornet_small_7x7()
            print('Loading:', 'hornet small-7')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/HorNet/hornet_small_7x7.pth')
        elif model_scale == 'small-gf':
            backbone = hornet_small_gf()
            print('Loading:', 'hornet small-gf')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/HorNet/hornet_small_gf.pth')
        else:
            sys.exit(model_scale + " is not a valid model scale! Hornet currently supported model scales are 'tiny-7' ,'tiny-gf', 'small-7', 'small-gf'.")
    
    if model_type == 'inceptionnext':
        if model_scale == 'tiny':
            backbone = inception_next_tiny()
            print('Loading:', 'inceptionnext tiny')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/InceptionNeXt/inceptionnext_tiny.pth')
        elif model_scale == 'small':
            backbone = inception_next_small()
            print('Loading:', 'inceptionnext small')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/InceptionNeXt/inceptionnext_small.pth')
        else:
            sys.exit(model_scale + " is not a valid model scale! Inceptionnext currently supported model scales are 'tiny' and 'small'.")

    if model_type == 'repvit':
        if model_scale == 'm11-300e':
            backbone = repvit_m1_1()
            print('Loading:', 'repvit m11-300e')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/RepViT/repvit_m1_1_distill_300e.pth')
        elif model_scale == 'm11-450e':
            backbone = repvit_m1_1()
            print('Loading:', 'repvit m11-450e')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/RepViT/repvit_m1_1_distill_450e.pth')
        elif model_scale == 'm15-300e':
            backbone = repvit_m1_5()
            print('Loading:', 'repvit m15-300e')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/RepViT/repvit_m1_5_distill_300e.pth')
        elif model_scale == 'm15-450e':
            backbone = repvit_m1_5()
            print('Loading:', 'repvit m15-450e')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/RepViT/repvit_m1_5_distill_450e.pth')
        else:
            sys.exit(model_scale + " is not a valid model scale! Repvit currently supported model scales are 'm11-300e' , 'm11-450e', 'm15-300e', 'm15-450e'.")
    
    if model_type == 'swintransformer':
        if model_scale == 'tiny':
            backbone = swin_tiny_patch4_window7_224()
            print('Loading:', 'swintransformer tiny')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/SwinTransformer/swin_tiny_patch4_window7_224.pth')
        elif model_scale == 'small':
            backbone = swin_small_patch4_window7_224()
            print('Loading:', 'swintransformer small')
            state_dict = torch.load('/Storage/share/wwrrgg/deformableLKA/2D/pretrained_pth/SwinTransformer/swin_small_patch4_window7_224.pth')
        else:
            sys.exit(model_scale + " is not a valid model scale! Swintransformer Currently supported model scales are 'tiny' and 'small'.")





    else:
        sys.exit(model_type + " is not a valid model scale! Currently supported model scales are  and  .")

    backbone.load_state_dict(state_dict, strict=False)
    print('Pretrain weights loaded.')

    return backbone
