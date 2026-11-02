"""Frozen MARS encoder with explicit study/sequence indexing and missing masks."""
import hashlib
from pathlib import Path

import torch

CHECKPOINT_SHA256 = '88388a59d92935a15d30bd41c0e9dbece99ba78db73cb374d44290fd6566ccc3'


def load_encoder(checkpoint):
    import monai
    from monai.networks.nets.swin_unetr import SwinTransformer
    if monai.__version__ not in {'1.4.0', '1.5.1'}:
        raise ValueError('MARS adapter requires a verified MONAI runtime')
    downsample = 'merging'
    if monai.__version__ == '1.5.1':
        # The newer default changes sampling offsets while preserving state
        # shapes. Preserve the original checkpoint computation explicitly.
        from mars_legacy_merging import MarsLegacyPatchMerging
        downsample = MarsLegacyPatchMerging
    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('MARS checkpoint identity changed')
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not all(k.startswith('swinViT.') for k in state):
        raise ValueError('Unexpected MARS checkpoint wrapper')
    model = SwinTransformer(in_chans=1, embed_dim=48, window_size=(7, 7, 7),
                            patch_size=(2, 2, 2), depths=(2, 2, 2, 2),
                            num_heads=(3, 6, 12, 24), mlp_ratio=4., qkv_bias=True,
                            norm_layer=torch.nn.LayerNorm, use_checkpoint=False,
                            spatial_dims=3, use_v2=True, downsample=downsample)
    model.load_state_dict({k.removeprefix('swinViT.'): v for k, v in state.items()}, strict=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def encode_study_sequences(encoder, volumes, present, chunk_size=1):
    """Return B,S,C,D,H,W grids; ignore missing slots and preserve study order.

    volumes must already follow an explicitly reviewed preprocessing contract.
    This function does not establish a DICOM preprocessing or accuracy claim.
    """
    if volumes.ndim != 6 or volumes.shape[2] != 1 or volumes.shape[-3:] != (96, 96, 96):
        raise ValueError('Expected B,S,1,96,96,96 volumes')
    if present.shape != volumes.shape[:2] or present.dtype != torch.bool:
        raise ValueError('Expected a boolean B,S acquisition mask')
    if present.device != volumes.device or not bool(present.any(dim=1).all()):
        raise ValueError('Each study needs a present acquisition on the input device')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if encoder.training:
        raise ValueError('Frozen extraction requires eval mode')
    b, s = present.shape
    # Flatten B,S together; never concatenate sequence-major then reshape B,S.
    flat = volumes.reshape(b*s, *volumes.shape[2:])
    indices = present.reshape(-1).nonzero(as_tuple=False).flatten()
    output = None
    for start in range(0, len(indices), chunk_size):
        ix = indices[start:start+chunk_size]
        selected = flat.index_select(0, ix)
        if not torch.isfinite(selected).all():
            raise ValueError('Nonfinite pixels in a present acquisition')
        features = encoder(selected)[4]
        if features.shape != (len(ix), 768, 3, 3, 3) or not torch.isfinite(features).all():
            raise ValueError('Unexpected or nonfinite MARS feature grid')
        if output is None:
            output = features.new_zeros((b*s, 768, 3, 3, 3))
        output.index_copy_(0, ix, features)
    return output.reshape(b, s, 768, 3, 3, 3)
