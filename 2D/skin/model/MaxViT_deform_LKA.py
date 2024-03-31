import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from .deformable_LKA.U_decoder import MBConv, ScaleDeformConv, DilationDeformConv
from .segformer import *
# from segformer import *

##################################
#
# LKA Modules
#
##################################

from timm.models.layers import DropPath



class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        #                       * self.attn(self.norm1(x)))

        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        #                       * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        # print("LKA return shape: {}".format(x.shape))
        return x




##################################
#
# End of LKA Modules
#
##################################


class FinalPatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = self.expand(x.permute(0, 2, 3, 1))

        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale,
            c=C, h=H, w=W
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())
        x = rearrange(x, "b (h w) c-> b c h w", h=H * self.dim_scale, w=W * self.dim_scale)  # B, C, H, W

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, image_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.image_size = image_size
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Up(nn.Module):
    def __init__(self, in_channels, pd=1, bilinear=True, linear=True):
        super(Up, self).__init__()
        self.linear = linear
        self.sd_conv = ScaleDeformConv(in_channels=in_channels, kernel_size=(2 * pd + 1, 2 * pd + 1), padding=pd)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        if linear:
            self.up_channel_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
            self.up_channel_2 = nn.Linear(in_channels, in_channels // 2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        if self.linear:
            x1 = self.up_channel_1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # b, h, w, c -> b, c, h, w
        else:
            x1 = self.conv1(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.sd_conv(x)

        if self.linear:
            x = self.up_channel_2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # b, h, w, c -> b, c, h, w
        else:
            x = self.conv2(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, num_class=9):
        super(OutConv, self).__init__()
        self.linear = nn.Linear(in_channels, in_channels)
        self.PE = FinalPatchExpand(dim=in_channels)
        self.last_layer = nn.Conv2d(in_channels, num_class, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.PE(x)
        x = self.last_layer(x)
        return x


##########################################
#
# MaxViT stuff
#
##########################################
# from merit_lib.networks import MaxViT4Out_Small, MaxViT4Out_Small3D
from model.merit_lib.networks import MaxViT4Out_Small
from model.merit_lib.networks import MaxViT4Out


class MaxViT_Tiny_deformableLKAFormer(nn.Module):
    def __init__(self, num_classes=1, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder pretrained tiny
        self.encoder = MaxViT4Out(n_class=num_classes, img_size=224, model_scale='tiny')

        # Decoder
        in_out_chan = [
            [64, 64, 64, 64, 64],
            [128, 128, 128, 128, 128],
            [256, 256, 256, 256, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]

        self.my_decoder_0 = Up(in_channels=in_out_chan[3][0], pd=1, bilinear=True, linear=True)
        self.my_decoder_1 = Up(in_channels=in_out_chan[2][0], pd=2, bilinear=True, linear=True)
        self.my_decoder_2 = Up(in_channels=in_out_chan[1][0], pd=3, bilinear=True, linear=True)
        self.outConv = OutConv(in_channels=in_out_chan[0][0], num_class=num_classes)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.encoder(x)

        # print(output_enc_3.shape, output_enc_2.shape, output_enc_1.shape, output_enc_0.shape)
        # torch.Size([20, 512, 7, 7]) torch.Size([20, 256, 14, 14]) torch.Size([20, 128, 28, 28]) torch.Size([20, 64, 56, 56])
        # ---------------Decoder-------------------------
        b, c, _, _ = output_enc_3.shape
        temp_3 = self.my_decoder_0(output_enc_3, output_enc_2)
        temp_2 = self.my_decoder_1(temp_3, output_enc_1)
        temp_1 = self.my_decoder_2(temp_2, output_enc_0)
        temp_0 = self.outConv(temp_1)

        # torch.Size([20, 256, 14, 14]) torch.Size([20, 128, 28, 28]) torch.Size([20, 64, 56, 56]) torch.Size([20, 9, 224, 224])

        return temp_0


class MaxViT_Small_deformableLKAFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.encoder = MaxViT4Out_Small(n_class=num_classes, img_size=224, pretrain=False)

        # Decoder
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]

        self.my_decoder_0 = Up(in_channels=in_out_chan[3][0], bilinear=False, linear=False)
        self.my_decoder_1 = Up(in_channels=in_out_chan[2][0], bilinear=False, linear=False)
        self.my_decoder_2 = Up(in_channels=in_out_chan[1][0], bilinear=False, linear=False)
        self.outConv = OutConv(in_channels=in_out_chan[0][0], num_class=9)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.encoder(x)

        # print(output_enc_3.shape, output_enc_2.shape, output_enc_1.shape, output_enc_0.shape)
        # print(
        # "torch.Size([20, 768, 7, 7]) torch.Size([20, 384, 14, 14]) torch.Size([20, 192, 28, 28]) torch.Size([20, 96, 56, 56])")
        # ---------------Decoder-------------------------
        b, c, _, _ = output_enc_3.shape
        temp_3 = self.my_decoder_0(output_enc_3, output_enc_2)
        temp_2 = self.my_decoder_1(temp_3, output_enc_1)
        temp_1 = self.my_decoder_2(temp_2, output_enc_0)
        temp_0 = self.outConv(temp_1)

        # print(tmp_3.shape, tmp_2.shape, tmp_1.shape, tmp_0.shape)
        # torch.Size([20, 196, 384])
        # torch.Size([20, 784, 192])
        # torch.Size([20, 3136, 96])
        # torch.Size([20, 9, 224, 224])

        return temp_0
