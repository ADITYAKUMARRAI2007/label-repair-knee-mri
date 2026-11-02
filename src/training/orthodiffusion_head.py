"""Fixed 3-plane spatial/global classification head for frozen diffusion features."""
import math
import torch
from torch import nn


class DiffusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders were pretrained separately; learn a separate map from each
        # feature basis before attention combines their spatial cells.
        self.project=nn.ModuleList([nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU())
                                    for _ in range(3)])
        self.plane=nn.Parameter(torch.randn(3,256)*.01)
        self.query=nn.Parameter(torch.randn(12,256)*.01)
        self.weight=nn.Parameter(torch.randn(12,512)*.01)
        self.bias=nn.Parameter(torch.zeros(12))
        self.dropout=nn.Dropout(.1)
        axis=torch.linspace(-1,1,16)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        frequency=torch.exp(-math.log(10000)*torch.arange(64)/64)
        x=x.flatten()[:,None]*frequency[None];y=y.flatten()[:,None]*frequency[None]
        coordinates=.02*torch.cat([x.sin(),x.cos()-1,y.sin(),y.cos()-1],dim=1)
        self.register_buffer('coordinates',coordinates)

    def forward(self,x,mask):
        if x.ndim!=4 or x.shape[1:]!=(3,256,256):
            raise ValueError('expected batch x3planes x256cells x256channels')
        if mask.shape!=x.shape[:2] or mask.dtype!=torch.bool or not mask.any(1).all():
            raise ValueError('invalid or empty plane mask')
        token=torch.stack([project(x[:,plane].float()) for plane,project in enumerate(self.project)],dim=1)
        token=token+self.plane[None,:,None]+self.coordinates[None,None]
        token=token.flatten(1,2)
        valid=mask[:,:,None].expand(-1,-1,256).flatten(1,2)
        score=torch.einsum('bkd,td->btk',token,self.query)/16
        attention=score.masked_fill(~valid[:,None],-torch.inf).softmax(-1)
        local=torch.einsum('btk,bkd->btd',attention,token)
        global_mean=(token*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True)
        fused=torch.cat([local,global_mean[:,None].expand(-1,12,-1)],dim=-1)
        return (self.dropout(fused)*self.weight[None]).sum(-1)+self.bias
