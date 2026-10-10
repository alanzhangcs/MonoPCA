from lib.models.monopca.backbones.eva02.main import EVA02
from lib.models.monopca.backbones.resnet.backbone import build_resnet_backbone


def build_vfm(configs):
    vfm_backbone = EVA02(**configs)
    return vfm_backbone


def build_resnet(configs):
    resnet_backbone = build_resnet_backbone(**configs)
    return resnet_backbone
