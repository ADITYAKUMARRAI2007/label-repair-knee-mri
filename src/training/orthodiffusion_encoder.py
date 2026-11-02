"""Pinned author encoder with explicit noise for reproducible feature extraction."""
import ast
import hashlib
import math
import sys
from functools import partial
from inspect import isfunction
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_encoder(source, checkpoint, expected_sha):
    assert sha(checkpoint) == expected_sha, 'checkpoint identity mismatch'
    sys.path.insert(0, str(source))
    from diffusion_model.unet import create_model
    trainer = Path(source) / 'diffusion_model/trainer.py'
    names = {'exists', 'default', 'extract', 'noise_like',
             'cosine_beta_schedule', 'GaussianDiffusion'}
    nodes = [n for n in ast.parse(trainer.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    env = dict(torch=torch, nn=nn, F=F, np=np, math=math, partial=partial,
               isfunction=isfunction, autocast=torch.cuda.amp.autocast)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(trainer), 'exec'), env)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = {k.replace('denoise_fn.module.', 'denoise_fn.', 1)
             if k.startswith('denoise_fn.module.') else k: v for k, v in payload['ema'].items()}
    assert len(state) == len(payload['ema']) == 257
    assert all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values())
    with torch.device('meta'):
        unet = create_model(256, 64, 1, in_channels=1, out_channels=1)
    model = env['GaussianDiffusion'](unet, image_size=256, depth_size=16,
                                   timesteps=1000, loss_type='l2')
    result = model.load_state_dict(state, strict=True, assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval().requires_grad_(False)
    return model


def feature_with_noise(model, volume, noise, timestep=30, amp=False):
    """Same q_sample + unconditioned mid_2 path as original get_feature."""
    assert not model.with_condition and not model.class_cond
    assert volume.shape == noise.shape and volume.device == noise.device
    t = torch.full((len(volume),), timestep, dtype=torch.long, device=volume.device)
    noisy = model.q_sample(x_start=volume, t=t, noise=noise)
    with torch.autocast(device_type=volume.device.type, dtype=torch.float16, enabled=amp):
        _, info = model.denoise_fn(noisy, t, ret_in=False, ret_mid=True, ret_out=False)
    return info['mid']['mid_2']


def synthetic_volume(size):
    z = torch.linspace(-1, 1, 16)[:, None, None]
    y = torch.linspace(-1, 1, size)[None, :, None]
    x = torch.linspace(-1, 1, size)[None, None, :]
    return (.4 * torch.sin(2*x) + .3 * torch.cos(3*y) + .2*z)[None, None].contiguous()
