"""Released OrthoFoundation forward, with fixed CLS + mean-patch pooling.

Architecture code must be provided from the pinned offline DINOv3 bundle.
This pooling is our transfer experiment, not the author's downstream recipe.
"""
import torch
from torch import nn


class OrthoEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        # Import before meta context: Dynamo initialization probes real CPU ops.
        from dinov3.models.vision_transformer import DinoVisionTransformer
        from dinov3.hub.backbones import dinov3_vitl16
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if len(state) != 368 or not all(k.startswith('backbone.') for k in state):
            raise RuntimeError('unexpected released checkpoint structure')
        with torch.device('meta'):
            self.backbone = dinov3_vitl16(pretrained=False)
        self.backbone.load_state_dict({k.removeprefix('backbone.'): v for k, v in state.items()},
                                      strict=True, assign=True)
        self.eval().requires_grad_(False)
        if sum(p.numel() for p in self.parameters()) != 303154176:
            raise RuntimeError('encoder parameter identity changed')

    def forward(self, x):
        if x.shape[1:] != (3, 224, 224):
            raise RuntimeError('expected normalized RGB 224 input')
        tokens, _ = self.backbone.prepare_tokens_with_masks(x)
        # The released author's full-finetune wrapper omits the RoPE argument.
        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        if tokens.shape[1:] != (201, 1024):
            raise RuntimeError('expected CLS, four register tokens and 196 patches')
        return torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=1)


def normalize_pixels(gray):
    """Input float grayscale in [0,255], batch x1 x224 x224."""
    mean = gray.new_tensor([.485, .456, .406])[None, :, None, None]
    std = gray.new_tensor([.229, .224, .225])[None, :, None, None]
    return (gray.div(255).expand(-1, 3, -1, -1) - mean) / std
