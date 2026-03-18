import torch
import torch.nn as nn
import numbers
from einops import rearrange

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

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)  ##返回所有元素的方差
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MLP2D(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super(MLP2D, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x_seq = x.view(B, C, H * W).permute(0, 2, 1)
        identity = x_seq
        x_seq = self.norm(x_seq)
        x_seq = self.fc1(x_seq)
        x_seq = self.act(x_seq)
        x_seq = self.drop1(x_seq)
        x_seq = self.fc2(x_seq)
        x_seq = self.drop2(x_seq)
        x_seq = x_seq + identity
        x_img = x_seq.permute(0, 2, 1).view(B, -1, H, W)

        return x_img


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
        return self.relu(x)

class Decoder(nn.Module):
    def __init__(self, dim, scale_factor=2, outdim2=32):
        super(Decoder, self).__init__()
        self.down = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.dwconv1 = DWConv(dim // 2, dim // 2, dilation=1)
        self.dwconv3 = DWConv(dim // 2, dim // 2, dilation=3)
        self.dwconv5 = DWConv(dim // 2, dim // 2, dilation=5)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = MLP2D(in_channels=dim // 2, out_channels=dim // 2)
        self.bn = nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv_out1 = DWConv(dim // 2, dim // 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.out2 = nn.Conv2d(dim // 2, outdim2, kernel_size=1)
        self.conv_out2 = BasicConv2d(outdim2, outdim2, kernel_size=3, padding=1)

        self.out = nn.Conv2d(outdim2, 1, kernel_size=1)

    def forward(self, x):
        x_down = self.down(x)

        x1 = self.dwconv1(x_down)
        x3 = self.dwconv3(x_down)
        x5 = self.dwconv5(x_down)
        x_fuse = x1 + x3 + x5

        x_refine = self.bn(x_fuse)
        x_refine = self.relu(x_refine)
        x_refine = self.dwconv_out1(x_refine)
        x_attn = x_refine + x_down

        x_ap = self.ap(x_attn)
        x_ap = self.mlp(x_ap)
        x_ap_flat = x_ap.flatten(2)
        x_ap_weight = self.sigmoid(x_ap_flat).reshape(x_ap.shape)
        x_attn = x_attn * x_ap_weight
        out2_feature = self.out2(x_attn)
        out2_refined = self.conv_out2(out2_feature)
        x_up_for_mask = self.up(out2_refined)
        M_dec1 = self.out(x_up_for_mask)

        return M_dec1, out2_feature
