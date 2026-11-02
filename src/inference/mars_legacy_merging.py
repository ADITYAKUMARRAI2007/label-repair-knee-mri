# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preserve MONAI1.4 patch merging for its pretrained MARS checkpoint.

Adapted from MONAI1.4.0 monai/networks/nets/swin_unetr.py PatchMerging.
MONAI1.5 changes three sampled offsets without changing parameter shapes.
Those historical duplicated offsets are necessary for checkpoint parity.
"""
import torch
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import PatchMergingV2


class MarsLegacyPatchMerging(PatchMergingV2):
    def forward(self, x):
        if x.ndim == 4:
            return super().forward(x)
        if x.ndim != 5:
            raise ValueError(f'Expected five-dimensional features, got {x.shape}')
        _, d, h, w, _ = x.shape
        if h % 2 or w % 2 or d % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        # Retain these exact offsets, including duplicates, from MONAI1.4.
        offsets = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        x = torch.cat([x[:, i::2, j::2, k::2, :] for i, j, k in offsets], dim=-1)
        return self.reduction(self.norm(x))
