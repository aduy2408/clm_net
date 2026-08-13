from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from natten import NeighborhoodAttention2D
from timm.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel, out_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, out_channel)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class LightweightGlobalBottleneck(nn.Module):
    """LGB: 1x1 channel reduction + depthwise refinement before global attention."""

    def __init__(self, in_channels, out_channels, bottleneck_channels=192, expand_ratio=2, num_heads=4):
        super().__init__()
        if bottleneck_channels % num_heads != 0:
            raise ValueError(f"bottleneck_channels {bottleneck_channels} must be divisible by {num_heads}")
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )
        self.depthwise = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            groups=bottleneck_channels,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(bottleneck_channels)
        self.attention = GlobalAttention(bottleneck_channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(bottleneck_channels)
        self.mlp = Mlp(bottleneck_channels, expand_ratio * bottleneck_channels, bottleneck_channels)
        self.project = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.reduce(x)
        x = self.depthwise(x) + x
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.attention(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.project(x)


class LightweightNeighborhoodAttentionBlock(nn.Module):
    """LNAB: apply neighborhood attention to a channel subset and keep the rest as identity."""

    def __init__(self, channels, attn_ratio=0.5, num_heads=3, kernel_size=3):
        super().__init__()
        if not 0.0 < attn_ratio <= 1.0:
            raise ValueError(f"attn_ratio must be in (0, 1], got {attn_ratio}")
        desired = int(round(channels * attn_ratio))
        attn_channels = max(8, int(round(desired / 8)) * 8)
        attn_channels = max(8, min(channels, attn_channels))
        heads = min(num_heads, max(1, attn_channels // 8))
        while attn_channels % heads != 0 or (attn_channels // heads) % 8 != 0:
            heads -= 1
            if heads < 1:
                raise ValueError(f"Cannot build LNAB for channels={channels}, attn_channels={attn_channels}")

        self.attn_channels = attn_channels
        self.skip_channels = channels - attn_channels
        self.pre = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm1 = nn.LayerNorm(attn_channels)
        self.attn = NeighborhoodAttention2D(attn_channels, num_heads=heads, kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(attn_channels)
        self.mlp = Mlp(attn_channels, 2 * attn_channels, attn_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.pre(x)
        x_attn, x_skip = torch.split(x, [self.attn_channels, self.skip_channels], dim=1)
        x_attn = x_attn.permute(0, 2, 3, 1).contiguous()
        attn = self.attn(self.norm1(x_attn)) + x_attn
        x_attn = self.mlp(self.norm2(attn)) + attn
        x_attn = x_attn.permute(0, 3, 1, 2).contiguous()
        if self.skip_channels:
            x_attn = torch.cat([x_attn, x_skip], dim=1)
        return self.fuse(x_attn)


class M3Skip(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convl = nn.Conv2d(in_channels[0], in_channels[1], 3, 2, 1)
        self.convm = nn.Conv2d(in_channels[1], in_channels[1], 3, 1, 1)
        self.convs = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels[2], in_channels[1], 3, 1, 1),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels[1], in_channels[1], 3, 1, 1),
            nn.BatchNorm2d(in_channels[1]),
            nn.GELU(),
        )

    def forward(self, xl, xm, xs):
        return self.fuse_conv(torch.cat([self.convl(xl), self.convm(xm), self.convs(xs)], dim=1))


class M2Skip(nn.Module):
    def __init__(self, in_channels, model_type="bottom"):
        super().__init__()
        self.model_type = model_type
        if model_type == "bottom":
            out = in_channels[1]
            self.convl = nn.Conv2d(in_channels[0], out, 3, 2, 1)
            self.convs = nn.Conv2d(in_channels[1], out, 3, 1, 1)
        else:
            out = in_channels[0]
            self.convl = nn.Conv2d(in_channels[0], out, 3, 1, 1)
            self.convs = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels[1], out, 3, 1, 1),
            )
        self.fuse_conv = nn.Sequential(nn.Conv2d(2 * out, out, 3, 1, 1), nn.BatchNorm2d(out), nn.GELU())

    def forward(self, xl, xs):
        return self.fuse_conv(torch.cat([self.convl(xl), self.convs(xs)], dim=1))


class PyramidPool(nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        h, w = x5.shape[-2:]
        xs = [
            F.adaptive_avg_pool2d(x1, (h, w)),
            F.adaptive_avg_pool2d(x2, (h, w)),
            F.adaptive_avg_pool2d(x3, (h, w)),
            F.adaptive_avg_pool2d(x4, (h, w)),
            x5,
        ]
        return torch.cat(xs, dim=1)


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class SSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class ReparamConv(nn.Module):
    def __init__(self, in_channels, expand_channels, out_channels, large_kernel_size, kernel_size, stride=1, deploy=False, se_kind="sse"):
        super().__init__()
        self.large_kernel_size = large_kernel_size
        self.kernel_size = kernel_size
        self.expand_channels = expand_channels
        self.stride = stride
        self.deploy = deploy
        self.se = SE(expand_channels) if se_kind == "se" else SSE(expand_channels)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, 1),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(inplace=True),
        )
        if deploy:
            self.fuse_conv = nn.Conv2d(expand_channels, expand_channels, large_kernel_size, stride, large_kernel_size // 2, groups=expand_channels, bias=True)
        else:
            self.large_conv = self._dw_branch(expand_channels, large_kernel_size, stride)
            self.square_conv = self._dw_branch(expand_channels, kernel_size, stride)
            self.ver_conv = self._dw_branch(expand_channels, (kernel_size, 1), stride, (kernel_size // 2, 0))
            self.hor_conv = self._dw_branch(expand_channels, (1, kernel_size), stride, (0, kernel_size // 2))
        self.active = nn.GELU()
        self.pointwise_conv = nn.Conv2d(expand_channels, out_channels, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    @staticmethod
    def _dw_branch(channels, kernel_size, stride, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=False)),
            ("bn", nn.BatchNorm2d(channels)),
        ]))

    def forward(self, x):
        x1 = self.expand_conv(x)
        if self.deploy:
            out = self.fuse_conv(x1)
        else:
            out = self.large_conv(x1) + self.square_conv(x1) + self.ver_conv(x1) + self.hor_conv(x1)
        return self.pointwise_conv(self.se(self.active(out))) + self.shortcut(x)


class PSUp(nn.Module):
    def __init__(self, in_channels, out_channels, r=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * r * r, 1)
        self.ps = nn.PixelShuffle(r)
        self.refine = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.refine(self.ps(self.proj(x)))
