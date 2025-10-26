import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=32, idx=None):#(B,N,C)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1) # (BNk)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :] # (BNk C)
    feature = feature.view(batch_size, num_points, k, num_dims) # (B N k C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # (B N k C)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (B 2C N k)
  
    return feature


class Semantic_Group(nn.Module):
    def __init__(self, output_channels=768, k=4): # 4
        super(Semantic_Group, self).__init__()
        self.outchannel = output_channels
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm1d(768)
        self.conv1 = nn.Sequential(nn.Conv2d(768*2, 256, kernel_size=1, bias=False),self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 768, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, pos): # x torch.Size([B, N, C])
        x = x+pos
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        x_neighbor= get_graph_feature(x, k=self.k) # (B 2C N k)
        x1 = self.conv1(x_neighbor) # (B 256 N k)  
        x1 = x1.max(dim=-1, keepdim=False)[0] 
        x = self.conv2(x1).permute(0,2,1)
        return x

class MLP(nn.Module):
    def __init__(self, output_channels=128):
        super(MLP, self).__init__()
        self.outchannel = output_channels
        self.conv1 = nn.Conv1d(3, self.outchannel, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.outchannel, self.outchannel, kernel_size=1, bias=False)
        self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x): # x torch.Size([B, 3, 8192])
        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        return x

