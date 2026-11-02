"""Twelve target queries attend to full frozen MARS spatial grids."""
import torch
from torch import nn


class MarsSpatialHead(nn.Module):
    def __init__(self, targets=12, hidden=256, dropout=.1):
        super().__init__()
        self.feature_norm = nn.LayerNorm(768)
        self.projection = nn.Linear(768, hidden)
        self.position = nn.Linear(3, hidden, bias=False)
        self.plane = nn.Embedding(3, hidden)
        self.queries = nn.Parameter(torch.empty(targets, hidden))
        self.attention = nn.MultiheadAttention(hidden, 8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.refine = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.score_weight = nn.Parameter(torch.empty(targets, hidden))
        self.score_bias = nn.Parameter(torch.zeros(targets))
        axis = torch.linspace(-1., 1., 3)
        coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
        self.register_buffer('coordinates', coordinates.reshape(27, 3))
        nn.init.normal_(self.queries, std=.02)
        nn.init.normal_(self.score_weight, std=.02)

    def forward(self, features, present):
        if features.ndim != 6 or features.shape[1:] != (3, 768, 3, 3, 3):
            raise ValueError('Expected B,3,768,3,3,3 frozen spatial features')
        if present.shape != features.shape[:2] or present.dtype != torch.bool:
            raise ValueError('Expected a boolean B,3 plane mask')
        if present.device != features.device or not bool(present.any(dim=1).all()):
            raise ValueError('Each study needs a present plane on the input device')
        if not bool(torch.isfinite(features[present]).all()):
            raise ValueError('Nonfinite features in a present plane')
        # Erase absent planes before normalization: masked NaNs cannot contaminate attention.
        clean = torch.where(present[:, :, None, None, None, None], features, 0.)
        tokens = clean.permute(0, 1, 3, 4, 5, 2).reshape(len(features), 3, 27, 768)
        tokens = self.projection(self.feature_norm(tokens))
        tokens = tokens + self.position(self.coordinates)[None, None]
        tokens = tokens + self.plane.weight[None, :, None]
        tokens = tokens.reshape(len(features), 81, -1)
        mask = (~present[:, :, None]).expand(-1, -1, 27).reshape(len(features), 81)
        query = self.queries[None].expand(len(features), -1, -1)
        attended, _ = self.attention(query, tokens, tokens, key_padding_mask=mask,
                                     need_weights=False)
        pooled = self.norm(query + attended)
        pooled = pooled + self.refine(pooled)
        return (pooled * self.score_weight[None]).sum(dim=-1) + self.score_bias
