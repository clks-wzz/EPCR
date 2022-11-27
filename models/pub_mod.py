import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class GCN_Att(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
        cha_ratio=8, spa_ratio=8):
        super(GCN_Att, self).__init__()
        down_ratio = 8
        self.in_channel = in_channel
        self.in_spatial = in_spatial
  
        self.use_spatial = use_spatial
        self.use_channel = use_channel

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        
        # Embedding functions for relation features
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel*2, out_channels=self.inter_channel,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        
        # Networks for learning attention weights
        if self.use_channel:    
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c//down_ratio,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_c//down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_c//down_ratio, out_channels=1,
                        kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                                kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                            kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        
    def forward(self, x):

        
        if self.use_channel:
            # channel attention
            # xc: Bx392x768x1
            xc = x.unsqueeze(-1)
            #print(xc.shape)
            # theta_xc: Bx768x49
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            #print(theta_xc.shape)
            # phi_xc: Bx49x768
            phi_xc = self.phi_channel(xc).squeeze(-1)
            #print(phi_xc.shape)
            # Gc: Bx768x768
            Gc = torch.matmul(theta_xc, phi_xc)
            #print(Gc.shape)
            # Gc_in: Bx768x768x1
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            #print(Gc_in.shape)
            # Gc_out: Bx768x768x1
            Gc_out = Gc.unsqueeze(-1)
            #print(Gc_out.shape)
            # Gc_joint: Bx1536x768x1
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            #print(Gc_joint.shape)
            # Gc_joint: Bx96x768x1
            Gc_joint = self.gg_channel(Gc_joint)
            #print(Gc_joint.shape)
            # g_xc: Bx49x768x1
            g_xc = self.gx_channel(xc)
            #print(g_xc.shape)
            # g_xc: Bx1x768x1
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            #print(g_xc.shape)
            # yc: Bx97x768x1
            yc = torch.cat((g_xc, Gc_joint), 1)
            #print(yc.shape)
            # W_yc: Bx768x1x1
            W_yc = self.W_channel(yc).squeeze(-1)
            #print(W_yc.shape)
            out = F.sigmoid(W_yc) * x
            return out