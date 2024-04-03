import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from .utils.U_decoder import MBConv, ScaleDeformConv, DilationDeformConv


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
from .networks.models.backbone import MaxViT_Out, Biformer_Out, ConvNext_Out, HorNet_Out, InceptionNext_Out, RepViT_Out, \
    SwinTransformer_Out


class MaxViT_Tiny_deformableLKAFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder pretrained tiny
        # self.encoder = RepViT_Out(n_class=num_classes, model_scale='m15e', pretrain=False)
        # self.encoder = MaxViT_Out(n_class=num_classes, model_scale='tiny', pretrain=True)
        # self.encoder = HorNet_Out(n_class=num_classes, model_scale='tiny-gf', pretrain=False)
        # self.encoder = Biformer_Out(n_class=num_classes, model_scale='small')

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
        self.outConv = OutConv(in_channels=in_out_chan[0][0], num_class=9)

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
    def __init__(self, model_type, model_scale, pretrain, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        self.model_scale = model_scale
        self.encoder = SwinTransformer_Out(n_class=num_classes, model_scale=model_scale,
                                           pretrain=True)  # Swin Transformer only can be used here, model scale: tiny, small
        # self.encoder = InceptionNext_Out(n_class=num_classes, model_scale='tiny', pretrain=True)
        # self.encoder = HorNet_Out(n_class=num_classes,model_scale='small-7', pretrain=True)
        # self.encoder = ConvNext_Out(n_class=num_classes, model_scale='small', pretrain=False)
        # self.encoder = MaxViT_Out(n_class=num_classes, model_scale='small', pretrain=True)

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

        # print(output_enc_3.shape, output_enc_2.shape, output_enc_1.shape, output_enc_0.shape, "from encoder")
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