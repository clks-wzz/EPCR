import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
import math
import numpy as np
from .hard_triplet_loss import HardTripletLoss


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def D_Dense(p, z, version='original'): # negative cosine similarity
    def _reshape(_in, _t=False):
        shape_b, shape_c, shape_h, shape_w = _in.shape
        if _t:
            _out = _in.permute(0,2,3,1).reshape(shape_b, shape_h * shape_w, shape_c)
        else:
            _out = _in.reshape(shape_b, shape_c, shape_h * shape_w)
        return _out

    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        p = _reshape(p, _t=True)
        z = _reshape(z, _t=False)
        loss = torch.bmm(p, z)

        return -loss.mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        print('Error implementation: have not implemented')
        #return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        raise ValueError
    else:
        raise Exception


def D_Dense_Batch(p, z, labels, needs=None, version='original'): # negative cosine similarity
    def _reshape(_in, _t=False):
        shape_b, shape_c, shape_h, shape_w = _in.shape
        if _t:
            _out = _in.permute(0,2,3,1).reshape(shape_b * shape_h * shape_w, shape_c)
        else:
            _out = _in.permute(0,2,3,1).reshape(shape_b * shape_h * shape_w, shape_c)
            _out = _out.permute(1, 0)
        return _out

    if version == 'original':
        shape_b, shape_c, shape_h, shape_w = p.shape

        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        p = _reshape(p, _t=True)
        z = _reshape(z, _t=False)
        mat_sim = torch.mm(p, z)

        mat_sim = mat_sim.reshape(shape_b, shape_h * shape_w, shape_b * shape_h * shape_w)
        mat_sim = mat_sim.reshape(shape_b, shape_h * shape_w, shape_b, shape_h * shape_w)
        mat_sim = torch.mean(mat_sim, dim=1)
        mat_sim = torch.mean(mat_sim, dim=-1)

        labels = labels.unsqueeze(1)
        mat_labels = torch.tensor((labels == labels.transpose(1, 0)).clone().detach(), dtype=torch.float32).cuda(needs.device)

        if needs is None:
            mat_needs = torch.ones(shape_b, shape_b)
        else:
            mat_needs = needs.unsqueeze(1).float()
            mat_needs = torch.mm(mat_needs, mat_needs.transpose(1, 0)) + torch.eye(shape_b).cuda(needs.device)
            mat_needs[torch.where(mat_needs > 0)] = 1.
        
        mat_mask = mat_labels * mat_needs

        #print('labels:', labels)
        #print('mat_labels:', mat_labels)
        #print('needs', needs)
        #print('mat_needs', mat_needs)

        loss = mat_sim * mat_mask

        return -loss.sum() / mat_mask.sum()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        print('Error implementation: have not implemented')
        #return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        raise ValueError
    else:
        raise Exception

def MSE(input, target):
    b, c, h, w = input.shape
    target = target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
    target = target.repeat(1, 1, h, w)
    loss = F.mse_loss(input, target, reduction='none')
    #print(loss.shape)
    loss = torch.mean(loss, dim=1, keepdim=True)
    loss = torch.mean(loss, dim=2, keepdim=True)
    loss = torch.mean(loss, dim=3, keepdim=True)
    loss = loss.squeeze()
    return loss

def MSE_SmoothLabel(input, target, detach=True):
    if detach:
        target = target.detach()
    loss = F.mse_loss(input, target, reduction='none')
    #print(loss.shape)
    loss = torch.mean(loss, dim=1, keepdim=True)
    loss = torch.mean(loss, dim=2, keepdim=True)
    loss = torch.mean(loss, dim=3, keepdim=True)
    loss = loss.squeeze()
    return loss

def LogRegression(input, target):
    b, c, h, w = input.shape
    target = target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
    target = target.repeat(1, 1, h, w)
    loss = -target * torch.log(input) - (1.-target) * torch.log(1.-input)
    loss = torch.mean(loss, dim=2, keepdim=True)
    loss = torch.mean(loss, dim=3, keepdim=True)
    loss = loss.squeeze()
    return loss

def LogRegression_SmoothLabel(input, target):
    target = target.detach()
    loss = -target * torch.log(input) - (1.-target) * torch.log(1.-input)
    loss = torch.mean(loss, dim=2, keepdim=True)
    loss = torch.mean(loss, dim=3, keepdim=True)
    loss = loss.squeeze()
    return loss


@torch.cuda.amp.autocast()  # fp16
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes,
                       kernel_size=1,
                       groups=1,
                       stride=1,
                       padding=0,
                       bias=False,
                       use_relu=True,
                       use_norm=True):
        super(BasicBlock, self).__init__()
        self.use_relu = use_relu
        self.use_norm = use_norm
        self.conv = nn.Conv2d(inplanes, planes,
                              kernel_size=kernel_size,
                              groups=groups,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        if self.use_norm:
            self.bn = nn.BatchNorm2d(planes)
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class conv2d_1x1(nn.Module):
    def __init__(self, inplanes, planes, groups=1, use_relu=False, use_norm=False):
        super(conv2d_1x1, self).__init__()
        self.block = BasicBlock(inplanes, planes, kernel_size=1, groups=groups, stride=1, padding=0, bias=False, use_relu=use_relu, use_norm=use_norm)
    def forward(self, x):
        x = self.block(x)
        return x

class conv2d_3x3(nn.Module):
    def __init__(self, inplanes, planes, groups=1, use_relu=False, use_norm=False):
        super(conv2d_3x3, self).__init__()
        self.block = BasicBlock(inplanes, planes, kernel_size=3, groups=groups, stride=1, padding=1, bias=False, use_relu=use_relu, use_norm=use_norm)
    def forward(self, x):
        x = self.block(x)
        return x

class conv2d_5x5(nn.Module):
    def __init__(self, inplanes, planes, groups=1, use_relu=False, use_norm=False):
        super(conv2d_5x5, self).__init__()
        self.block = BasicBlock(inplanes, planes, kernel_size=5, groups=groups, stride=1, padding=2, bias=False, use_relu=use_relu, use_norm=use_norm)
    def forward(self, x):
        x = self.block(x)
        return x


class Causal_Norm_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, use_effect=True, num_head=2, tau=16.0, alpha=1.0, gamma=0.03125, mu=0.9, *args):
        super(Causal_Norm_Classifier, self).__init__()
        # default alpha = 3.0
        #self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_head
        self.use_effect = use_effect
        self.relu = nn.ReLU(inplace=True)
        self.mu = mu

        '''
        self.hidden_layer = nn.Sequential(
            conv2d_1x1(feat_dim, self.feat_dim),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        '''

        self.register_parameter('weight', nn.Parameter(torch.Tensor(num_classes, self.feat_dim), requires_grad=True))
        self.register_buffer('embed_mean', torch.zeros([1, self.feat_dim, 1, 1]))

        self.reset_parameters(self.weight)
    
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        #torch.nn.init.constant_(weight, stdv)
    
    def conv_mm(self, feat, weight):
        b, c, h, w = feat.shape
        out_dim = weight.shape[-1]
        feat = feat.permute(0, 2, 3, 1).reshape(b*h*w, c)
        out = torch.mm(feat, weight)
        out = out.reshape(b, h, w, out_dim).permute(0, 3, 1, 2)
        return out

    def update_embed_mean(self, x):
        self.embed_mean.data = self.mu * self.embed_mean.data + (1 - self.mu) * x.detach().mean(0,keepdim=True).mean(2,keepdim=True).mean(3,keepdim=True).data

    def forward(self, x, training=True, remove_effect=False):
        #print(self.weight)
        #print(self.embed_mean.squeeze())
        #x = self.hidden_layer(x)

        if training:
            #self.embed_mean = self.mu * self.embed_mean + x.detach().mean(0,keepdim=True).mean(2,keepdim=True).mean(3,keepdim=True)
            self.update_embed_mean(x)
        #print('embed_mean:', self.embed_mean.squeeze())
        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        #print(normed_w)
        y = self.conv_mm(normed_x * self.scale, normed_w.t())
        #y = self.conv_mm(x, self.weight.t())

        # remove the effect of confounder c during test
        if not training and remove_effect:
            normed_c = self.multi_head_call(self.l2_norm, self.embed_mean.detach())
            #print(self.weight)
            #print(self.embed_mean.squeeze())
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = self.conv_mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                #print('nx:', nx.squeeze())
                #print('nw:', nw.squeeze())
                #print('cos_val:', cos_val)
                #print('nc:', nc.squeeze())
                #print('nw:', nw)
                #print('y0:', y0)
                output.append(y0)
            y = sum(output)
            #y = sum(output) / self.num_head
        
        y = (y + 1.) / 2.

        return y

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 4 or len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / (torch.norm(x, 2, 1, keepdim=True))
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, conv_ks=1):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        if conv_ks == 1:
            self.layer1 = nn.Sequential(
                conv2d_1x1(in_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = nn.Sequential(
                conv2d_1x1(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer3 = nn.Sequential(
                conv2d_1x1(hidden_dim, out_dim),
                nn.BatchNorm2d(out_dim)
            )
        elif conv_ks == 3:
            self.layer1 = nn.Sequential(
                conv2d_3x3(in_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = nn.Sequential(
                conv2d_3x3(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer3 = nn.Sequential(
                conv2d_3x3(hidden_dim, out_dim),
                nn.BatchNorm2d(out_dim)
            )
        else:
            raise ValueError
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048, conv_ks=1): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        if conv_ks == 1:
            self.layer1 = nn.Sequential(
                conv2d_1x1(in_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = conv2d_1x1(hidden_dim, out_dim)
        elif conv_ks == 3:
            self.layer1 = nn.Sequential(
                conv2d_3x3(in_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layer2 = conv2d_3x3(hidden_dim, out_dim)
        else:
            raise ValueError
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class classify_MLP(nn.Module):
    def __init__(self, in_dim=2048, n_classes=1, conv_ks=1): # bottleneck structure
        super().__init__()
        if conv_ks == 1:
            self.layer1 = nn.Sequential(
                conv2d_1x1(in_dim, n_classes),
            )
        elif conv_ks == 3:
            self.layer1 = nn.Sequential(
                conv2d_3x3(in_dim, n_classes),
            )
        else:
            raise ValueError

    def forward(self, x):
        x = self.layer1(x)
        return x 

class classify_MLP_Ensemble(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64*3, n_classes=1, ensemble_num=1): # bottleneck structure
        super().__init__()
        if hidden_dim > 0:
            self.layer1 = nn.Sequential(
                conv2d_1x1(in_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv2d_1x1(hidden_dim, n_classes, groups=ensemble_num),
            )
        else:
            self.layer1 = nn.Sequential(
                conv2d_1x1(in_dim, n_classes, groups=ensemble_num),
            )

        #self.pool2d = F.adaptive_avg_pool2d
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if len(x.shape) == 4:
            if x.shape[2] * x.shape[3] > 1:
                x = self.pool2d(x)
        x = self.layer1(x)
        x = x.squeeze()
        return x 

class classifyPred_MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=3, n_classes=1): # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            conv2d_5x5(in_dim, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv2d_3x3(hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x 


class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py
    #@staticmethod
    def forward(self, input):
        self.iter_num += 1
        return input * 1.0
    #@staticmethod
    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

#class GRL_FP16(torch.autograd.Function):
#    def __init__(self):
#        self.iter_num = 0
#        self.alpha = 10
#        self.low = 0.0
#        self.high = 1.0
#        self.max_iter = 4000  # be same to the max_iter of config.py
#    #@staticmethod
#    def forward(self, input):
#        self.iter_num += 1
#        return input * 1.0
#    #@staticmethod
#    def backward(self, gradOutput):
#        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
#                         - (self.high - self.low) + self.low)
#        return -coeff * gradOutput

class GRL_FP16(torch.autograd.Function):
    iter_num = 0
    alpha = 10
    low_value = 0.0
    high_value = 1.0
    max_iter = 4000

    @staticmethod
    def forward(ctx, input):
        GRL_FP16.iter_num += 1
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        coeff = np.float(
            2.0 * (GRL_FP16.high_value - GRL_FP16.low_value) /
            (1.0 + np.exp(-GRL_FP16.alpha * GRL_FP16.iter_num / GRL_FP16.max_iter))
            - (GRL_FP16.high_value - GRL_FP16.low_value) + GRL_FP16.low_value
        )
        return - coeff * grad_output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(64, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2,
            nn.Softmax(dim=1)
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1)
        adversarial_out = self.ad_net(self.grl_layer(feature))
        #adversarial_out = self.ad_net(feature)
        return adversarial_out


class Discriminator_FP16(nn.Module):
    def __init__(self):
        super(Discriminator_FP16, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(64, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2,
            nn.Softmax(dim=1)
        )
        self.grl_layer = GRL_FP16()

    @torch.cuda.amp.autocast()
    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1)
        #adversarial_out = self.ad_net(self.grl_layer(feature))
        adversarial_out = self.ad_net(self.grl_layer.apply(feature))
        return adversarial_out

'''v11_SSDG normal classifier ''' 
class SimSiam_Semi_CDCN_MeanTeacherV11_SSDG(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        assert backbone is not None
        
        self.backbone = backbone
        self.pool2d = F.adaptive_avg_pool2d
        self.discriminator = Discriminator()
        self.projector = projection_MLP(backbone.output_dim, 64, 64)

        #self.encoder = nn.Sequential( # f encoder
        #    self.backbone,
        #    self.projector
        #)
        self.predictor = prediction_MLP(in_dim=64, hidden_dim=64, out_dim=64)
        self.classifier = classify_MLP(in_dim=64, n_classes=1)

        self.loss_triplet = HardTripletLoss(margin=0.1, hardest=False)
        self.loss_softmax = nn.CrossEntropyLoss(reduction='none')
    
    def postprocess_map(self, maps):
        v = torch.mean(maps, dim=2, keepdim=True)
        v = torch.mean(v, dim=3, keepdim=True)
        v = v.squeeze()
        if v.shape == torch.Size([]):
            v = v.unsqueeze(0).unsqueeze(1)
        else:
            v = v.unsqueeze(1)
        #print(v.shape)
        v_final = torch.cat([1.-v, v], dim=1)
        return v_final

    def inference_train(self, x1, x2, label, domains, triplets):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        e1, e2 = self.backbone(x1)['embbed'], self.backbone(x2)['embbed']

        p, h = self.projector, self.predictor
        z1, z2 = p(e1), p(e2)
        p1, p2 = h(z1), h(z2)
        loss_us = D_Dense(p1, z2) / 2 + D_Dense(p2, z1) / 2

        '''supervised training part'''
        c = self.classifier
        pz1, pz2 = c(z1), c(z2)
        pc1, pc2 = c(p1), c(p2)
        
        # SSDG parts
        d1_avg = self.pool2d(z1, (1,1))
        d1_avg = d1_avg.squeeze()
        '''
        d1 = self.discriminator(d1_avg)
        #print(d1_avg.shape)
        real_adloss = self.loss_softmax(d1, domains)
        real_adloss = real_adloss.squeeze() * (1 - label) 
        real_adloss = torch.sum(real_adloss) / torch.sum(1 - label)
        '''
        triplet_loss = self.loss_triplet(d1_avg, triplets)

        #loss_ssdg = 1.0 * triplet_loss + 0.1 * real_adloss
        loss_ssdg = triplet_loss

        # FOCR-FAS parts
        ce1 = MSE(pz1, label) / 2. + MSE(pz2, label) / 2.
        ce1_smooth = MSE_SmoothLabel(pc1, pz2, detach=False) / 2. + MSE_SmoothLabel(pc2, pz1, detach=False) / 2.

        #loss_ce = torch.mean(loss_ce)
        '''
        loss_ce = torch.sum(ce1) / (torch.sum(needs) + 1.)
        loss_smooth = torch.mean(ce1_smooth)
        loss_ce += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce
        '''
        loss_ce = torch.mean(ce1)
        loss_smooth = torch.mean(ce1_smooth)
        loss_us += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce + loss_ssdg

        losses = {
            'd1_avg': d1_avg,
            'loss_us': loss_us,
            'loss_ce': loss_ce,
            'loss_ssdg': loss_ssdg,
            'loss_tt': loss_tt
        }
        
        return losses, [self.postprocess_map(pz1), self.postprocess_map(pz2)]

    def inference_test(self, x1):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        b1 = self.backbone(x1)['embbed']

        p, h, c = self.projector, self.predictor, self.classifier

        z1 = p(b1)

        #p1 = h(z1)
        
        #c1 = c(p1, False, True)
        #c1 = c(p1, False, False)
        c1 = c(z1)


        c1 = self.postprocess_map(c1)

        return c1


    def forward(self, x1, x2=None, label=None, domains=None, triplets=None):
        if not x2 is None:
            return self.inference_train(x1, x2, label, domains, triplets)
        else:
            return self.inference_test(x1)


'''v11_SSDG normal classifier ''' 
class SimSiam_Semi_CDCN_MeanTeacherV11_SSDG_FP16(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        assert backbone is not None
        
        self.backbone = backbone
        self.pool2d = F.adaptive_avg_pool2d
        #self.discriminator = Discriminator()
        self.projector = projection_MLP(backbone.output_dim, 64, 64)

        #self.encoder = nn.Sequential( # f encoder
        #    self.backbone,
        #    self.projector
        #)
        self.predictor = prediction_MLP(in_dim=64, hidden_dim=64, out_dim=64)
        self.classifier = classify_MLP(in_dim=64, n_classes=1)

        self.loss_triplet = HardTripletLoss(margin=0.1, hardest=False)
        self.loss_softmax = nn.CrossEntropyLoss(reduction='none')
    
    def postprocess_map(self, maps):
        v = torch.mean(maps, dim=2, keepdim=True)
        v = torch.mean(v, dim=3, keepdim=True)
        v = v.squeeze()
        if v.shape == torch.Size([]):
            v = v.unsqueeze(0).unsqueeze(1)
        else:
            v = v.unsqueeze(1)
        #print(v.shape)
        v_final = torch.cat([1.-v, v], dim=1)
        return v_final

    def inference_train(self, x1, x2, label, domains, triplets):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        e1, e2 = self.backbone(x1)['embbed'], self.backbone(x2)['embbed']

        p, h = self.projector, self.predictor
        z1, z2 = p(e1), p(e2)
        p1, p2 = h(z1), h(z2)
        loss_us = D_Dense(p1, z2) / 2 + D_Dense(p2, z1) / 2

        '''supervised training part'''
        c = self.classifier
        pz1, pz2 = c(z1), c(z2)
        pc1, pc2 = c(p1), c(p2)
        
        # SSDG parts
        d1_avg = self.pool2d(z1, (1,1))
        d1_avg = d1_avg.squeeze()
        '''
        d1 = self.discriminator(d1_avg)
        #print(d1_avg.shape)
        real_adloss = self.loss_softmax(d1, domains)
        real_adloss = real_adloss.squeeze() * (1 - label) 
        real_adloss = torch.sum(real_adloss) / torch.sum(1 - label)
        '''
        triplet_loss = self.loss_triplet(d1_avg, triplets)

        #loss_ssdg = 1.0 * triplet_loss + 0.1 * real_adloss
        loss_triplet = triplet_loss

        # FOCR-FAS parts
        ce1 = MSE(pz1, label) / 2. + MSE(pz2, label) / 2.
        ce1_smooth = MSE_SmoothLabel(pc1, pz2, detach=False) / 2. + MSE_SmoothLabel(pc2, pz1, detach=False) / 2.

        #loss_ce = torch.mean(loss_ce)
        '''
        loss_ce = torch.sum(ce1) / (torch.sum(needs) + 1.)
        loss_smooth = torch.mean(ce1_smooth)
        loss_ce += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce
        '''
        loss_ce = torch.mean(ce1)
        loss_smooth = torch.mean(ce1_smooth)
        loss_us += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce

        losses = {
            'd1_avg': d1_avg,
            'loss_us': loss_us,
            'loss_ce': loss_ce,
            'loss_triplet': loss_triplet,
            'loss_tt': loss_tt
        }
        
        return losses, [self.postprocess_map(pz1), self.postprocess_map(pz2)]

    def inference_test(self, x1):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        b1 = self.backbone(x1)['embbed']

        p, h, c = self.projector, self.predictor, self.classifier

        z1 = p(b1)

        #p1 = h(z1)
        
        #c1 = c(p1, False, True)
        #c1 = c(p1, False, False)
        c1 = c(z1)


        c1 = self.postprocess_map(c1)

        return c1

    @torch.cuda.amp.autocast()
    def forward(self, x1, x2=None, label=None, domains=None, triplets=None):
        if not x2 is None:
            return self.inference_train(x1, x2, label, domains, triplets)
        else:
            return self.inference_test(x1)

'''v12_SSDG normal classifier ''' 
class SimSiam_Semi_CDCN_MeanTeacherV12_SSDG_FP16(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        assert backbone is not None
        
        self.backbone = backbone
        self.pool2d = F.adaptive_avg_pool2d
        #self.discriminator = Discriminator()
        self.projector = projection_MLP(backbone.output_dim, 64, 64)

        #self.encoder = nn.Sequential( # f encoder
        #    self.backbone,
        #    self.projector
        #)
        self.predictor = prediction_MLP(in_dim=64, hidden_dim=64, out_dim=64)
        self.classifier = classify_MLP_Ensemble(in_dim=64, n_classes=2)

        self.loss_triplet = HardTripletLoss(margin=0.1, hardest=False)
        self.loss_softmax = nn.CrossEntropyLoss(reduction='none')
    
    def postprocess_map(self, maps):
        '''
        v = torch.mean(maps, dim=2, keepdim=True)
        v = torch.mean(v, dim=3, keepdim=True)
        v = v.squeeze()
        if v.shape == torch.Size([]):
            v = v.unsqueeze(0).unsqueeze(1)
        else:
            v = v.unsqueeze(1)
        #print(v.shape)
        v_final = torch.cat([1.-v, v], dim=1)
        '''
        v_final = maps
        return v_final

    def inference_train(self, x1, x2, label, domains, triplets):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        e1, e2 = self.backbone(x1)['embbed'], self.backbone(x2)['embbed']

        p, h = self.projector, self.predictor
        z1, z2 = p(e1), p(e2)
        p1, p2 = h(z1), h(z2)
        loss_us = D_Dense(p1, z2) / 2 + D_Dense(p2, z1) / 2

        '''supervised training part'''
        c = self.classifier
        pz1, pz2 = c(z1), c(z2)
        pc1, pc2 = c(p1), c(p2)
        
        # SSDG parts
        d1_avg = self.pool2d(z1, (1,1))
        d1_avg = d1_avg.squeeze()
        '''
        d1 = self.discriminator(d1_avg)
        #print(d1_avg.shape)
        real_adloss = self.loss_softmax(d1, domains)
        real_adloss = real_adloss.squeeze() * (1 - label) 
        real_adloss = torch.sum(real_adloss) / torch.sum(1 - label)
        '''
        triplet_loss = self.loss_triplet(d1_avg, triplets)

        #loss_ssdg = 1.0 * triplet_loss + 0.1 * real_adloss
        loss_triplet = triplet_loss

        # FOCR-FAS parts
        #ce1 = MSE(pz1, label) / 2. + MSE(pz2, label) / 2.
        ce1 = self.loss_softmax(pz1, label) / 2. + self.loss_softmax(pz2, label) / 2.
        #ce1_smooth = MSE_SmoothLabel(pc1, pz2, detach=False) / 2. + MSE_SmoothLabel(pc2, pz1, detach=False) / 2.
        ce1_smooth = softmax_kl_loss(pc1, pz2) / 2. + softmax_kl_loss(pc2, pz1) / 2.
        #softmax_kl_loss

        #loss_ce = torch.mean(loss_ce)
        '''
        loss_ce = torch.sum(ce1) / (torch.sum(needs) + 1.)
        loss_smooth = torch.mean(ce1_smooth)
        loss_ce += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce
        '''
        loss_ce = torch.mean(ce1)
        loss_smooth = torch.mean(ce1_smooth)
        loss_us += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce

        losses = {
            'd1_avg': d1_avg,
            'loss_us': loss_us,
            'loss_ce': loss_ce,
            'loss_triplet': loss_triplet,
            'loss_tt': loss_tt
        }
        
        return losses, [self.postprocess_map(pz1), self.postprocess_map(pz2)]

    def inference_test(self, x1):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        b1 = self.backbone(x1)['embbed']

        p, h, c = self.projector, self.predictor, self.classifier

        z1 = p(b1)

        #p1 = h(z1)
        
        #c1 = c(p1, False, True)
        #c1 = c(p1, False, False)
        c1 = c(z1)


        c1 = self.postprocess_map(c1)

        return c1

    @torch.cuda.amp.autocast()
    def forward(self, x1, x2=None, label=None, domains=None, triplets=None):
        if not x2 is None:
            return self.inference_train(x1, x2, label, domains, triplets)
        else:
            return self.inference_test(x1)


'''v11_SSDG normal classifier ''' 
class SimSiam_Semi_CDCN_MeanTeacherV13_SSDG_FP16(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        assert backbone is not None
        
        self.backbone = backbone
        self.pool2d = F.adaptive_avg_pool2d
        #self.discriminator = Discriminator()
        self.projector = projection_MLP(backbone.output_dim, 64, 64)

        #self.encoder = nn.Sequential( # f encoder
        #    self.backbone,
        #    self.projector
        #)
        self.predictor = prediction_MLP(in_dim=64, hidden_dim=64, out_dim=64)
        self.classifier = classify_MLP(in_dim=64, n_classes=1)

        self.loss_triplet = HardTripletLoss(margin=0.1, hardest=False)
        self.loss_softmax = nn.CrossEntropyLoss(reduction='none')
    
    def postprocess_map(self, maps):
        v = torch.mean(maps, dim=2, keepdim=True)
        v = torch.mean(v, dim=3, keepdim=True)
        v = v.squeeze()
        if v.shape == torch.Size([]):
            v = v.unsqueeze(0).unsqueeze(1)
        else:
            v = v.unsqueeze(1)
        #print(v.shape)
        v_final = torch.cat([1.-v, v], dim=1)
        return v_final

    def inference_train(self, x1, x2, label, domains, triplets):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        e1, e2 = self.backbone(x1)['embbed'], self.backbone(x2)['embbed']

        p, h = self.projector, self.predictor
        z1, z2 = p(e1), p(e2)
        p1, p2 = h(z1), h(z2)
        loss_us = D_Dense(p1, z2) / 2 + D_Dense(p2, z1) / 2

        '''supervised training part'''
        c = self.classifier
        pz1, pz2 = c(z1), c(z2)
        pc1, pc2 = c(p1), c(p2)
        
        # SSDG parts
        d1_avg = self.pool2d(z1, (1,1))
        d1_avg = d1_avg.squeeze()
        '''
        d1 = self.discriminator(d1_avg)
        #print(d1_avg.shape)
        real_adloss = self.loss_softmax(d1, domains)
        real_adloss = real_adloss.squeeze() * (1 - label) 
        real_adloss = torch.sum(real_adloss) / torch.sum(1 - label)
        '''
        triplet_loss = self.loss_triplet(d1_avg, triplets)

        #loss_ssdg = 1.0 * triplet_loss + 0.1 * real_adloss
        loss_triplet = triplet_loss

        # FOCR-FAS parts
        ce1 = MSE(pz1, label) / 2. + MSE(pz2, label) / 2.
        ce1_smooth = MSE_SmoothLabel(pc1, pz2, detach=False) / 2. + MSE_SmoothLabel(pc2, pz1, detach=False) / 2.

        #loss_ce = torch.mean(loss_ce)
        '''
        loss_ce = torch.sum(ce1) / (torch.sum(needs) + 1.)
        loss_smooth = torch.mean(ce1_smooth)
        loss_ce += 0.1 * loss_smooth

        loss_tt = loss_us + loss_ce
        '''
        loss_ce = torch.mean(ce1)
        loss_smooth = torch.mean(ce1_smooth)
        loss_us += 0.1 * loss_smooth

        loss_tt = 0.01 * loss_us + loss_ce

        losses = {
            'd1_avg': d1_avg,
            'loss_us': loss_us,
            'loss_ce': loss_ce,
            'loss_triplet': loss_triplet,
            'loss_tt': loss_tt
        }
        
        return losses, [self.postprocess_map(pz1), self.postprocess_map(pz2)]

    def inference_test(self, x1):
        #print(x1.shape, x2.shape, label.shape, needs.shape)
        '''self training part'''
        b1 = self.backbone(x1)['embbed']

        p, h, c = self.projector, self.predictor, self.classifier

        z1 = p(b1)

        #p1 = h(z1)
        
        #c1 = c(p1, False, True)
        #c1 = c(p1, False, False)
        c1 = c(z1)


        c1 = self.postprocess_map(c1)

        return c1

    @torch.cuda.amp.autocast()
    def forward(self, x1, x2=None, label=None, domains=None, triplets=None):
        if not x2 is None:
            return self.inference_train(x1, x2, label, domains, triplets)
        else:
            return self.inference_test(x1)

if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












