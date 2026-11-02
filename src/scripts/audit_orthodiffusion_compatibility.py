#!/usr/bin/env python3
"""Strict-load one pinned original encoder and verify seeded feature replay on CPU."""
import ast
import argparse
import hashlib
import json
import math
import signal
import sys
import time
from functools import partial
from inspect import isfunction
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/knee_pretraining/orthodiffusion'
SOURCE=OUT/'source'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-size',type=int,choices=[64,256],default=256)
    args=parser.parse_args()
    fixture_size=args.fixture_size
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('180-second compatibility cap exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    receipt=json.loads((OUT/'source_receipt.json').read_text())
    for item in receipt['files']:assert sha(SOURCE/item['path'])==item['sha256']
    checkpoint=json.loads((OUT/'checkpoint_receipt.json').read_text())
    path=ROOT/checkpoint['file'];assert sha(path)==checkpoint['sha256']
    sys.path.insert(0,str(SOURCE))
    from diffusion_model.unet import create_model
    # Read the original class and its arithmetic helpers without importing its
    # unrelated data loaders, visualization dependencies or training side effects.
    trainer_path=SOURCE/'diffusion_model/trainer.py'
    parsed=ast.parse(trainer_path.read_text())
    names={'exists','default','extract','noise_like','cosine_beta_schedule','GaussianDiffusion'}
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected}==names
    env={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'partial':partial,
         'isfunction':isfunction,'autocast':torch.cuda.amp.autocast}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(trainer_path),'exec'),env)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert isinstance(payload,dict) and isinstance(payload.get('ema'),dict)
    state={k.replace('denoise_fn.module.','denoise_fn.',1) if k.startswith('denoise_fn.module.') else k:v
           for k,v in payload['ema'].items()}
    assert len(state)==len(payload['ema']) and all(torch.is_tensor(v) for v in state.values())
    with torch.device('meta'):
        unet=create_model(256,64,1,in_channels=1,out_channels=1)
    model=env['GaussianDiffusion'](unet,image_size=256,depth_size=16,timesteps=1000,loss_type='l2')
    expected=model.state_dict()
    missing=sorted(set(expected)-set(state));extra=sorted(set(state)-set(expected))
    mismatches={k:{'expected':list(expected[k].shape),'observed':list(v.shape)} for k,v in state.items()
                if k in expected and expected[k].shape!=v.shape}
    structure={'checkpoint_sha256':checkpoint['sha256'],'payload_keys':list(payload),
               'tensor_count':len(state),'missing':missing,'unexpected':extra,'shape_mismatches':mismatches,
               'finite_tensors':all(bool(torch.isfinite(v).all()) for v in state.values()),
               'parameter_count':sum(p.numel() for p in model.parameters())}
    (OUT/'checkpoint_structure.json').write_text(json.dumps(structure,indent=2)+'\n')
    assert not missing and not extra and not mismatches and structure['finite_tensors']
    model.load_state_dict(state,strict=True,assign=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    # A reduced fixture checks forward mechanics/replay, not the full clinical input.
    z=torch.linspace(-1,1,16)[:,None,None]
    y=torch.linspace(-1,1,fixture_size)[None,:,None]
    x=torch.linspace(-1,1,fixture_size)[None,None,:]
    volume=(.4*torch.sin(2*x)+.3*torch.cos(3*y)+.2*z)[None,None].contiguous()
    timestep=torch.tensor([30],dtype=torch.long)
    outputs=[];durations=[]
    with torch.inference_mode():
        for attempt in range(2):
            torch.manual_seed(20261016)
            begin=time.monotonic()
            feature=model.get_feature(volume,timestep,name='mid_2',use_amp=False)
            durations.append(time.monotonic()-begin)
            assert torch.isfinite(feature).all()
            outputs.append(feature.clone())
    error=float((outputs[0]-outputs[1]).abs().max())
    assert error==0 and outputs[0].std()>0
    np.savez_compressed(OUT/f'cpu_fixture_{fixture_size}.npz',volume=volume.numpy(),
                        pooled=outputs[0].mean((2,3,4)).numpy(),
                        feature_probe=outputs[0][...,::4,::4].numpy())
    result={'status':('PASS_STRICT_LOAD_AND_SEEDED_AUTHOR_FEATURE_REPLAY' if fixture_size==256 else 'PASS_STRICT_LOAD_AND_REDUCED_SYNTHETIC_REPLAY_ONLY'),
            **structure,'input_shape':list(volume.shape),'feature_shape':list(outputs[0].shape),
            'forward_seconds':durations,'seeded_replay_max_error':error,
            'timestep':30,'block':'mid_2','seed':20261016,
            'noise_note':'Author get_feature draws Gaussian noise. Resetting a fixed seed makes this fixture repeatable; a dataset-order-independent per-series seed is required for real inference.',
            'author_input_note':'Contiguous central16 slices resized256; existing sparse44-slice study cache is not an equivalent volume.',
            'clinical_accuracy_measured':False,'gpu_runtime_measured':False,
            'full_author_input_size_validated':fixture_size==256,
            'source_commit':receipt['commit'],'audit_source_sha256':sha(Path(__file__)),
            'elapsed_seconds':time.monotonic()-started}
    (OUT/f'compatibility_audit_{fixture_size}.json').write_text(json.dumps(result,indent=2)+'\n')
    signal.alarm(0);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'compatibility_failure.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e)},indent=2)+'\n')
        raise
