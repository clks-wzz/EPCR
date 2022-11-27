import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .pub_mod import *
from .vision_transformer import deit_tiny_patch16_224, deit_small_patch16_224



class TransFAS(nn.Module):
    def __init__(self, model_type, pretrained=True):
        super().__init__()
        self.pretrained=pretrained
        self.build_model(model_type)
        self.output_dim = 64

    def build_model(self, model_type):
        if model_type == "small":
            self.encoder = deit_small_patch16_224(self.pretrained)
            dim_seq = 384
        elif model_type == "tiny":
            self.encoder = deit_tiny_patch16_224(self.pretrained)
            dim_seq = 192
        self.head = nn.Linear(dim_seq, 2)

        self.gcn_attention_list = nn.ModuleList([GCN_Att(392,  dim_seq) for i in range(3)])

        main_model_ = []
        # dim_seqx14x14 -> 128x28x28
        main_model_ += [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(dim_seq, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ]

        # 128x28x28 -> 64x28x28
        main_model_ += [
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]

        # 64x28x28 -> 1x28x28
        '''
        main_model_ += [
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        ]
        '''
        self.regressor = nn.Sequential(*[
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        ])

        self.decoder = nn.Sequential(*main_model_)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        B_, C_, W_, H_ = x.shape
        label=0
        leaf_fea0, leaf_fea1, leaf_fea2, leaf_fea3, x_cls = self.encoder(x)
        x_cls = self.head(x_cls)
        if label == 0:
            fea2 = self.gcn_attention_list[2](torch.cat((leaf_fea2, leaf_fea3), 1).permute(0, 2, 1))
            fea2 = fea2.permute(0, 2, 1)
            fea2_ = fea2[:,:196, :]+fea2[:, 196:, :]
            fea1 = self.gcn_attention_list[1](torch.cat((fea2_, leaf_fea1), 1).permute(0, 2, 1))
            fea1 = fea1.permute(0, 2, 1)
            fea1_ = fea1[:, :196, :]+fea1[:, 196:, :]
            fea0 = self.gcn_attention_list[0](torch.cat((fea1_, leaf_fea0), 1).permute(0, 2, 1))
            fea0 = fea0.permute(0, 2, 1)
            fea0_ = fea0[:, :196, :]+fea0[:, 196:, :]
            fea = fea0_
        elif label == 1:
            fea2 = self.gcn_attention_list[2](torch.cat((leaf_fea2, leaf_fea3), 1).permute(0, 2, 1))
            fea2 = fea2.permute(0, 2, 1)
            fea2_ = fea2[:,:196, :]+fea2[:, 196:, :]
            fea0 = self.gcn_attention_list[0](torch.cat((leaf_fea0, leaf_fea1), 1).permute(0, 2, 1))
            fea0 = fea0.permute(0, 2, 1)
            fea0_ = fea0[:, :196, :]+fea0[:, 196:, :]
            fea1 = self.gcn_attention_list[1](torch.cat((fea0_, fea2_), 1).permute(0, 2, 1))
            fea1 = fea1.permute(0, 2, 1)
            fea1_ = fea1[:, :196, :]+fea1[:, 196:, :]
            fea = fea1_
        elif label == 2:
            fea0 = self.gcn_attention_list[0](torch.cat((leaf_fea0, leaf_fea1), 1).permute(0, 2, 1))
            fea0 = fea0.permute(0, 2, 1)
            fea0_ = fea0[:,:196, :]+fea0[:, 196:, :]
            fea1 = self.gcn_attention_list[1](torch.cat((leaf_fea1, fea0_), 1).permute(0, 2, 1))
            fea1 = fea1.permute(0, 2, 1)
            fea1_ = fea1[:, :196, :]+fea1[:, 196:, :]
            fea2 = self.gcn_attention_list[2](torch.cat((leaf_fea2, fea1_), 1).permute(0, 2, 1))
            fea2 = fea2.permute(0, 2, 1)
            fea2_ = fea2[:, :196, :]+fea2[:, 196:, :]
            fea = fea2_



        elif label == 3:
            fea = leaf_fea0 + leaf_fea1 + leaf_fea2 + leaf_fea3
        elif label == 4:
            fea = leaf_fea0
        elif label == 5:
            fea = leaf_fea1
        elif label == 6:
            fea = leaf_fea2
        fea = fea.reshape(B_, 14, 14, -1).permute(0, 3, 1, 2)
        # pyramid_fea0 = x_fea_list[0]
        # pyramid_fea1 = x_fea_list[1]+pyramid_fea0
        # pyramid_fea2 = x_fea_list[2]+pyramid_fea1
        # pyramid_fea3 = x_fea_list[3]+pyramid_fea2
        fea_0 = self.decoder(fea)
        out = self.regressor(fea_0)

        features = {
            'map': out,
            'embbed': fea_0
        }
        
        #return out[:, 0, :, :], x_cls
        return features



if __name__ == '__main__':
    model = TransFAS("tiny").cuda()
    inputs = torch.ones(2, 3, 224, 224).cuda()
    #outputs, x_cls = model(inputs)
    #print(outputs.shape)
    #print(x_cls.shape)
    features = model(inputs)
    print(features['map'].shape)
    print(features['embbed'].shape)