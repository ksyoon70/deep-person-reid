from __future__ import division, absolute_import
import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    r"""ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    
    Reference:
        Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
        
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        s (float): scale factor.
        m (float): margin.
    """
    def __init__(self, num_classes, feat_dim, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.Tensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feats, labels):
        # 1. Normalize features and weights
        cosine = F.linear(F.normalize(feats), F.normalize(self.weight))
        
        # 2. Add margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Keep things numerically stable
        # if cosine > cos(pi - m), use phi. Else use cosine - m * sin(pi - m) (Taylor approx) to avoid jumps?
        # The paper suggests this condition: when theta > pi - m
        # In practice, many implementations effectively simplify this. 
        # Here we follow a standard robust implementation.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 3. Create one-hot encoding for labels to apply margin only to ground truth
        # (batch_size, num_classes)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 4. Final logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output
