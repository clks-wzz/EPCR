from .simsiam_semi_cdcn_meanteacher import SimSiam_Semi_CDCN_MeanTeacherV11
from .simsiam_semi_cdcn_meanteacher_SSDG import SimSiam_Semi_CDCN_MeanTeacherV11_SSDG, SimSiam_Semi_CDCN_MeanTeacherV11_SSDG_FP16
from torchvision.models import resnet50, resnet18
import torch
from .backbones import *

def get_backbone(backbone, castrate=True):
    backbone_name = backbone
    backbone = eval(f"{backbone}()")   

    if castrate:
        if 'resnet18_fas' in backbone_name:
            pass
        elif 'resnet' in backbone_name:
            backbone.output_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()
        elif 'mobilenet' in backbone_name:
            backbone.output_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()
        elif 'trans' in backbone_name:
            if 'transfas' not in backbone_name:
                backbone.head = torch.nn.Identity()

    return backbone


def get_model(name, backbone, args=None):
    if name == 'simsiam_semi_cdcn_meanteacherV11':
        model =  SimSiam_Semi_CDCN_MeanTeacherV11(get_backbone(backbone, castrate=False))
    elif name == 'simsiam_semi_cdcn_meanteacherV11_ssdg_fp16':
        model =  SimSiam_Semi_CDCN_MeanTeacherV11_SSDG_FP16(get_backbone(backbone, castrate=False))
    else:
        raise NotImplementedError
    return model






