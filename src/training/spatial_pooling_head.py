"""Shared head for a fixed global-versus-spatial feature comparison."""
import math

import torch
import torch.nn as nn


class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Same trainable parameters and initialization order as the prior head.
        self.project = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048,256), nn.GELU())
        self.slot = nn.Parameter(torch.randn(5,256)*.01)
        self.position = nn.Parameter(torch.randn(4,256)*.01)
        self.query = nn.Parameter(torch.randn(12,256)*.01)
        self.weight = nn.Parameter(torch.randn(12,256)*.01)
        self.bias = nn.Parameter(torch.zeros(12))
        self.dropout = nn.Dropout(.1)

    def coordinates(self, cells, device, dtype):
        if cells == 1:
            return torch.zeros(1,256,device=device,dtype=dtype)
        if cells != 49:
            raise RuntimeError('expected complete7x7 feature grid')
        axis = torch.linspace(-1,1,7,device=device,dtype=dtype)
        y,x = torch.meshgrid(axis,axis,indexing='ij')
        frequency = torch.exp(-math.log(10000)*torch.arange(64,device=device,dtype=dtype)/64)
        x = x.flatten()[:,None]*frequency[None]
        y = y.flatten()[:,None]*frequency[None]
        return .02*torch.cat([x.sin(),x.cos()-1,y.sin(),y.cos()-1],dim=1)

    def forward(self,x,mask):
        if x.ndim == 3:
            x = x.unsqueeze(2)
        if x.ndim != 4 or x.shape[1] != 20 or x.shape[-1] != 2048:
            raise RuntimeError('expected batch x20 x spatial cells x2048')
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool or not mask.any(1).all():
            raise RuntimeError('invalid or all-empty slice mask')
        cells = x.shape[2]
        token = self.project(x.float()).view(-1,5,4,cells,256)
        token = (token+self.slot[None,:,None,None]+self.position[None,None,:,None]
                 +self.coordinates(cells,token.device,token.dtype)[None,None,None]).flatten(1,3)
        valid = mask[:,:,None].expand(-1,-1,cells).flatten(1,2)
        score = torch.einsum('bkd,td->btk',token,self.query)/16
        attention = score.masked_fill(~valid[:,None],-torch.inf).softmax(-1)
        pooled = torch.einsum('btk,bkd->btd',attention,token)
        return (self.dropout(pooled)*self.weight[None]).sum(-1)+self.bias
