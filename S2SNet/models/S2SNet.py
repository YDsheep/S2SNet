import torch
import torch.nn as nn
from models.VMamba.classification.models import build_vssm_model
from models.VMamba.classification.config import get_config
from options import opt
import os
from models.Dec3L import Decoder
from models.decoder.dec import SSU
from models.CMMamba.Cross_Model_MambaFU import PatchEmbed, SinMamba_

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1):
        super(DWConv, self).__init__()
        self.out_planes = out_planes
        self.dwconv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=dilation,
                                groups=in_planes, dilation=dilation)
        self.pconv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pconv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class ECA(nn.Module):
    """ 高效通道注意力 """

    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

def channel_shuffle(x, groups):
    """标准的 Channel Shuffle 操作"""
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class RefineOptimized(nn.Module):
    def __init__(self, num, dims):
        super(RefineOptimized, self).__init__()
        self.num = num

        self.local_conv = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1, groups=num, bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True)
        )

        self.spatial_dwconv = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims, bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True)
        )

        self.conv_fu = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(dims)
        )

        self.eca = ECA(dims)

    def forward(self, x):
        residual = x
        out = self.local_conv(x)
        out = self.spatial_dwconv(out)
        out = channel_shuffle(out, self.num)
        out = self.conv_fu(out)
        out = self.eca(out)

        return out + residual

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class DualFreqECA(nn.Module):
    def __init__(self, k_size=3):
        super(DualFreqECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv_avg = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_max = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.fusion = nn.Conv1d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.fusion.weight, 0.5)

    def forward(self, x):
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C) -> (B, 1, C)
        y_avg = self.avg_pool(x).squeeze(-1).squeeze(-1).unsqueeze(1)
        y_max = self.max_pool(x).squeeze(-1).squeeze(-1).unsqueeze(1)

        y_avg = self.conv_avg(y_avg)  # (B, 1, C)
        y_max = self.conv_max(y_max)  # (B, 1, C)

        # (B, 2, C) -> (B, 1, C)
        y_cat = torch.cat([y_avg, y_max], dim=1)
        y = self.fusion(y_cat)

        # (B, 1, C) -> (B, C) -> (B, C, 1, 1)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)

        weight = self.sigmoid(y)

        return x * weight


class DynamicLocalConv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.conv5 = nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False)

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.att(x)  # [B, 2, 1, 1]
        return w[:, 0:1] * self.conv3(x) + w[:, 1:2] * self.conv5(x)


class DAR(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.reduce = nn.Conv2d(in_channel, out_channel, 1, bias=False)

        self.branch1 = nn.Sequential(
            DynamicLocalConv(out_channel),
            DualFreqECA(k_size=3)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1), groups=out_channel, bias=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0), groups=out_channel, bias=False),
            DualFreqECA(k_size=3)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, groups=out_channel, bias=False)
        )

        self.res = nn.Identity() if in_channel == out_channel else nn.Conv2d(in_channel, out_channel, 1, bias=False)

    def forward(self, x):
        residual = self.res(x)
        x = self.reduce(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)

        fused = self.fusion(torch.cat([b1, b2], dim=1))

        return fused + residual

class S2SNet(nn.Module):
    def __init__(self, channel=48):
        super(S2SNet, self).__init__()
        self.vmamba_config = get_config(opt)
        self.encoder = build_vssm_model(self.vmamba_config)
        if opt.pre:
              if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.encoder.load_state_dict(checkpoint['model'], strict=False)
                print("=> loaded checkpoint")

        self.reduce_sal1 = DAR(96, channel)
        self.reduce_sal2 = DAR(96, channel)
        self.reduce_sal3 = DAR(192, channel)
        self.reduce_sal4 = DAR(384, channel)
        self.reduce_sal5 = DAR(768, channel)

        self.re1 = RefineOptimized(2, channel)
        self.re2 = RefineOptimized(2, channel)
        self.re3 = RefineOptimized(2, channel)
        self.re4 = RefineOptimized(2, channel)
        self.re5 = RefineOptimized(2, channel)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            PatchEmbed(in_chans=channel, embed_dim=channel, patch_size=1, stride=1),
            SinMamba_(channel)
        )
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.out1 = Decoder(channel, 4)

        self.decoder2 = SSU(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.out2 = Decoder(channel, 4)

        self.decoder3 = SSU(channel)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.out3 = Decoder(channel, 8)

        self.decoder4 = SSU(channel)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.out4 = Decoder(channel, 16)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)


    def forward(self, x_rgb):

        x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb = self.encoder(x_rgb)
        x1_rgb = x1_rgb.permute(0, 3, 1, 2)
        x2_rgb = x2_rgb.permute(0, 3, 1, 2)
        x3_rgb = x3_rgb.permute(0, 3, 1, 2)
        x4_rgb = x4_rgb.permute(0, 3, 1, 2)
        x5_rgb = x5_rgb.permute(0, 3, 1, 2)

        x_sal1 = self.reduce_sal1(x1_rgb)
        x_sal2 = self.reduce_sal2(x2_rgb)
        x_sal3 = self.reduce_sal3(x3_rgb)
        x_sal4 = self.reduce_sal4(x4_rgb)
        x_sal5 = self.reduce_sal5(x5_rgb)

        sal5 = self.re5(x_sal5)
        sal4 = self.re4(x_sal4)
        sal3 = self.re3(x_sal3)
        sal2 = self.re2(x_sal2)
        sal1 = self.re1(x_sal1)

        x_4 = self.decoder4(sal5, sal4)
        x4_pred, _ = self.out4(self.conv4(x_4) + sal4)

        x_3 = self.decoder3(x_4, sal3)
        x3_pred, _ = self.out3(self.conv3(x_3) + sal3)

        x_2 = self.decoder2(x_3, sal2)
        x2_pred, _ = self.out2(self.conv2(x_2) + sal2)

        x1 = torch.cat([x_2, sal1], dim=1)
        x_1 = self.decoder1(x1)
        x1_pred, _ = self.out1(self.conv1(x_1) + sal1)

        return x1_pred, x2_pred, x3_pred, x4_pred


    
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
