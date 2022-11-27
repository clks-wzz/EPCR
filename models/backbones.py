from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from .cdcn import CDCN, CDCN_Pure, CDCN_FP16, CDCN_GN, CDCN_GN_Half, CDCN_Half, CDCNV0, CDCN_CDPool, CDCNpp, CDCNpp_FP16, CDCNpp_GN_FP16, CDCNpp_Half_FP16
from .resnet import _resnet18_fas, _resnet18_fas_sp
from .TransFAS import TransFAS

try:
    from vit_pytorch import ViT
except:
    print('Fail to import vit_pytorch')


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet50w2(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


# def resnet50w4(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


# def resnet50w5(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3])

def resnet18hifi(**kwargs):
    return resnet18Hifi(high_resolution=True)

def resnet18_fas(**kwargs):
    return _resnet18_fas(**kwargs)

def resnet18_fas_sp(**kwargs):
    return _resnet18_fas_sp(**kwargs)

def mobilenetv2(**kwargs):
    return MobileNetV2(last_channel=128)

def transfas(**kwargs):
    return TransFAS("tiny")


def auxiliary(**kwargs):
    return Auxiliary()

def cdcn(**kwargs):
    return CDCN()

def cdcn_pure(**kwargs):
    return CDCN_Pure()

def cdcn_fp16(**kwargs):
    return CDCN_FP16()

def cdcnV0(**kwargs):
    return CDCNV0()

def cdcn_half(**kwargs):
    return CDCN_Half()

def cdcn_cdpool(**kwargs):
    return CDCN_CDPool()

def cdcn_gn(**kwargs):
    return CDCN_GN()

def cdcn_gn_half(**kwargs):
    return CDCN_GN_Half()

def cdcnpp_fp16(**kwargs):
    return CDCNpp_FP16()

def cdcnpp_gn_fp16(**kwargs):
    return CDCNpp_GN_FP16()

def cdcnpp_half_fp16(**kwargs):
    return CDCNpp_Half_FP16()

