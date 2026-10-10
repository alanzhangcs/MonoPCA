import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.

    Computes a channel-wise gate from global average pooled features and
    re-weights the input accordingly.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        # Global pooling -> bottleneck -> gate.
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class RegionSegHead(nn.Module):
    """
    Multi-scale region segmentation head.

    Given a list of feature maps (typically a feature pyramid), this head:
    - builds a top-down fused pyramid `p` (FPN-style),
    - predicts a per-pixel region probability map at each level,
    - uses the probabilities to enhance the original features,
    - emits a hard/binary mask embedding per level (thresholded at 0.5).

    Returns:
        enhanced_features: list[Tensor], each same shape as input features[i]
        region_probs: list[Tensor], each shape (B, num_classes, Hi, Wi) in (0, 1)
        seg_embed: list[Tensor], binary masks (0/1) same shape as region_probs[i]
    """
    def __init__(self, n_levels=4, d_model=256, num_classes=1):
        super(RegionSegHead, self).__init__()
        input_proj_list = []
        pred_list = []
        for _ in range(n_levels):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )
            )
            pred_list.append(
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(d_model, num_classes, 3, 1, 1),
                )
            )
        self.input_proj = nn.ModuleList(input_proj_list)
        self.pred = nn.ModuleList(pred_list)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.deconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2)
                for _ in range(n_levels - 1)
            ]
        )
        self.attention = nn.ModuleList([SEBlock(d_model) for _ in range(n_levels)])

    def forward(self, features):

        # Build a top-down pyramid: project each level then add upsampled higher level.
        p = [None] * len(features)
        p[-1] = self.input_proj[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            p[i] = self.input_proj[i](features[i]) + self.upsample(p[i + 1])

        region_probs = []
        for i in range(len(features)):
            # SE attention on fused pyramid features before region prediction.
            p[i] = self.attention[i](p[i])
            prob = torch.sigmoid(self.pred[i](p[i]))
            # Clamp away from 0/1 to avoid numerical issues in downstream losses/ops.
            region_probs.append(torch.clamp(prob, 1e-5, 1 - 1e-5))

        # Enhance original multi-scale features by the predicted region probabilities.
        enhanced_features = [feat * prob for feat, prob in zip(features, region_probs)]

        # Binary region embeddings (used as a lightweight segmentation cue).
        seg_embed = [
            torch.where(prob > 0.5, torch.ones_like(prob), torch.zeros_like(prob))
            for prob in region_probs
        ]

        return enhanced_features, region_probs, seg_embed
