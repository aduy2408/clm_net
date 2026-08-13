import torch.nn as nn

from .modules import (
    LightweightGlobalBottleneck,
    LightweightNeighborhoodAttentionBlock,
    M2Skip,
    M3Skip,
    PSUp,
    PyramidPool,
    ReparamConv,
)


def make_reparam(in_channels, expand_channels, out_channels, se_kind):
    return ReparamConv(in_channels, expand_channels, out_channels, 5, 3, se_kind=se_kind)


class CLM_Net(nn.Module):
    def __init__(
        self,
        channel,
        n_classes=2,
        filters=None,
        lgb_bottleneck=192,
        lnab_kind="partial",
        lnab_ratios=None,
        upsample_kind="bilinear_conv",
        se_kind="sse",
        deep_supervision=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.filters = filters or [24, 24, 48, 96, 192]
        lnab_ratios = lnab_ratios or [0.5, 0.5, 0.5, 0.5]
        f = self.filters

        self.conv1 = nn.Sequential(make_reparam(channel, f[1], f[0], se_kind), make_reparam(f[0], f[1], f[0], se_kind))
        self.down1 = nn.Conv2d(f[0], f[1], 3, 2, 1)
        self.conv2 = nn.Sequential(make_reparam(f[1], f[2], f[1], se_kind), make_reparam(f[1], f[2], f[1], se_kind))
        self.down2 = nn.Conv2d(f[1], f[2], 3, 2, 1)
        self.conv3 = nn.Sequential(make_reparam(f[2], f[3], f[2], se_kind), make_reparam(f[2], f[3], f[2], se_kind))
        self.down3 = nn.Conv2d(f[2], f[3], 3, 2, 1)
        self.conv4 = nn.Sequential(make_reparam(f[3], f[4], f[3], se_kind), make_reparam(f[3], f[4], f[3], se_kind))
        self.down4 = nn.Conv2d(f[3], f[4], 3, 2, 1)

        self.dconv1 = nn.Sequential(make_reparam(f[3], f[4], f[3], se_kind), make_reparam(f[3], f[4], f[3], se_kind))
        self.dconv2 = nn.Sequential(make_reparam(f[2], f[3], f[2], se_kind), make_reparam(f[2], f[3], f[2], se_kind))
        self.dconv3 = nn.Sequential(make_reparam(f[1], f[2], f[1], se_kind), make_reparam(f[1], f[2], f[1], se_kind))
        self.dconv4 = nn.Sequential(make_reparam(f[0], f[1], f[0], se_kind), make_reparam(f[0], f[1], f[0], se_kind))

        self.pyramidpool = PyramidPool()
        self.lgb = LightweightGlobalBottleneck(sum(f), f[4], bottleneck_channels=lgb_bottleneck)

        self.up1 = self._upsample(upsample_kind, f[4], f[3])
        self.up2 = self._upsample(upsample_kind, f[3], f[2])
        self.up3 = self._upsample(upsample_kind, f[2], f[1])
        self.up4 = self._upsample(upsample_kind, f[1], f[0])

        self.skip1 = M2Skip([f[2], f[3]], "bottom")
        self.skip2 = M3Skip([f[1], f[2], f[3]])
        self.skip3 = M3Skip([f[0], f[1], f[2]])
        self.skip4 = M2Skip([f[0], f[1]], "top")

        if lnab_kind == "partial":
            self.lnab1 = LightweightNeighborhoodAttentionBlock(f[3], attn_ratio=lnab_ratios[0], num_heads=12)
            self.lnab2 = LightweightNeighborhoodAttentionBlock(f[2], attn_ratio=lnab_ratios[1], num_heads=6)
            self.lnab3 = LightweightNeighborhoodAttentionBlock(f[1], attn_ratio=lnab_ratios[2], num_heads=3)
            self.lnab4 = LightweightNeighborhoodAttentionBlock(f[0], attn_ratio=lnab_ratios[3], num_heads=3)
        elif lnab_kind == "identity":
            self.lnab1 = nn.Identity()
            self.lnab2 = nn.Identity()
            self.lnab3 = nn.Identity()
            self.lnab4 = nn.Identity()
        else:
            raise ValueError(f"Unsupported lnab_kind: {lnab_kind}")

        self.output_layer = nn.Conv2d(f[0], n_classes, 1)

    @staticmethod
    def _upsample(kind, in_channels, out_channels):
        if kind == "psup":
            return PSUp(in_channels, out_channels)
        if kind == "bilinear_conv":
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )
        raise ValueError(f"Unsupported upsample_kind: {kind}")

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        x_down4 = self.down4(x4)

        x5 = self.lgb(self.pyramidpool(x1, x2, x3, x4, x_down4))

        x46 = self.lnab1(self.skip1(x3, x4))
        x37 = self.lnab2(self.skip2(x2, x3, x4))
        x28 = self.lnab3(self.skip3(x1, x2, x3))
        x19 = self.lnab4(self.skip4(x1, x2))

        x6 = self.dconv1(self.up1(x5) + x46)
        x7 = self.dconv2(self.up2(x6) + x37)
        x8 = self.dconv3(self.up3(x7) + x28)
        x9 = self.dconv4(self.up4(x8) + x19)
        return self.output_layer(x9)
