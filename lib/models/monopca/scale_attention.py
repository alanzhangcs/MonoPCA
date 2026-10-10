import torch
import torch.nn as nn


class FeatureFusionModule(nn.Module):
    """
    Fuses two feature maps of the same shape with per-branch channel attention
    predicted from their concatenation.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        attention_weights = self.attention(torch.cat([feat1, feat2], dim=1))
        att1, att2 = attention_weights.chunk(2, dim=1)
        # channel-wise weighted sum of the two branches
        return feat1 * att1 + feat2 * att2


class AdaptiveFeatureFusion(nn.Module):
    """
    Multi-level adaptive feature fusion: a 1x1 adapter per branch and level,
    followed by an attention fusion module per level.
    """

    def __init__(self, channels, num_levels=3):
        super().__init__()
        self.num_levels = num_levels

        self.fusion_modules = nn.ModuleList([FeatureFusionModule(channels) for _ in range(num_levels)])

        self.resnet_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])

        self.eva_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])

    def forward(self, resnet_feat, eva_feat, level_idx):
        adapted_resnet = self.resnet_adapter[level_idx](resnet_feat)
        adapted_eva = self.eva_adapter[level_idx](eva_feat)
        return self.fusion_modules[level_idx](adapted_resnet, adapted_eva)
