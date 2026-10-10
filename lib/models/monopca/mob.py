import torch
import torch.nn as nn
from .scale_attention import AdaptiveFeatureFusion
from .position_encoding import PositionEmbeddingSine
from utils.misc import NestedTensor


class MoB(nn.Module):
    """ MoB (Mixture of Backbone): fuses multi-scale ResNet features with EVA-02 (ViT) features. """
    def __init__(self, resnet_backbone, vfm_backbone, hidden_dim=256,
                 num_channels=[512, 1024, 2048], num_feature_levels=4):
        super().__init__()

        if num_feature_levels > 1:
            num_backbone_outs = len(num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.resnet_backbone = resnet_backbone
        self.vfm_backbone = vfm_backbone
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.feature_fusion = AdaptiveFeatureFusion(channels=hidden_dim, num_levels=num_feature_levels)

    def forward(self, images):
        # vfm_features[0] is the extra high-resolution FPN level, not used for fusion
        vfm_features = self.vfm_backbone(images)[1:]
        resnet_features = self.resnet_backbone(images)

        srcs = []
        masks = []
        pos = []
        for l in range(self.num_feature_levels):
            # extra levels beyond the ResNet outputs are built from its last stage
            resnet_feat = resnet_features[min(l, len(resnet_features) - 1)]
            resnet_feat = self.input_proj[l](resnet_feat)
            src = self.feature_fusion(resnet_feat, vfm_features[l], l)
            mask = torch.zeros(src.shape[0], src.shape[2], src.shape[3], dtype=torch.bool, device=src.device)
            srcs.append(src)
            masks.append(mask)
            pos.append(self.position_embedding(NestedTensor(src, mask)))

        return srcs, masks, pos
