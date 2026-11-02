"""Check released weights and reproduce the author's reviewed forward on CPU."""
import ast
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
research=ROOT/'research/knee_pretraining'


def main():
    for name in ['orthofoundation','dinov3']:
        receipt=json.loads((research/name/'source_receipt.json').read_text())
        for entry in receipt['files']:
            path=research/name/'source'/entry['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('source changed: '+str(path))
    sys.path.insert(0,str(research/'dinov3/source'))
    torch.set_num_threads(2)
    # Load modules before entering a meta-device context: Dynamo import probes CPU ops.
    from dinov3.models.vision_transformer import DinoVisionTransformer
    from dinov3.hub.backbones import dinov3_vitl16
    before=time.monotonic()
    state=torch.load(ROOT/'models/orthofoundation/OrthoFoudation-L.pth',map_location='cpu',weights_only=True,mmap=True)
    if not all(k.startswith('backbone.') for k in state):raise RuntimeError('unexpected wrapper prefix')
    state={k.removeprefix('backbone.'):v for k,v in state.items()}
    with torch.device('meta'):model=dinov3_vitl16(pretrained=False)
    loaded=model.load_state_dict(state,strict=True,assign=True);model.eval().requires_grad_(False)
    if not all(torch.isfinite(v).all() for v in state.values()):raise RuntimeError('nonfinite checkpoint')
    source=research/'orthofoundation/source/nets/dinov3_vit.py'
    cls=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='ViT')
    # Extract only the reviewed class, preserving its forward; do not run SSL/network setup.
    namespace={'torch':torch,'nn':torch.nn,'copy':copy,'get_vit':lambda *a,**kw:model}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
    wrapper=namespace['ViT'](backbone='vitl16',mode='finetune',ft_nums=24).eval()
    gray=torch.linspace(0,1,224*224).reshape(1,1,224,224).expand(-1,3,-1,-1)
    x=(gray-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode():
        original=wrapper(x)[-1]
        tokens,_=model.prepare_tokens_with_masks(x)
        for block in model.blocks:tokens=block(tokens)
        reproduced=model.norm(tokens)
        native=model.forward_features(x)['x_norm_clstoken']
    if not torch.equal(original,reproduced) or not torch.isfinite(original).all():
        raise RuntimeError('author forward reproduction failed')
    result={'status':'STRICT_LOAD_AND_AUTHOR_FORWARD_PASS',
        'checkpoint_sha256':json.loads((research/'orthofoundation/checkpoint_receipt.json').read_text())['sha256'],
        'state_tensors':len(state),'all_finite':True,'missing_keys':loaded.missing_keys,
        'unexpected_keys':loaded.unexpected_keys,'parameters':sum(p.numel() for p in model.parameters()),
        'output_shape':list(original.shape),'author_forward_reproduction_max_abs_error':float((original-reproduced).abs().max()),
        'native_rope_vs_author_cls_max_abs_difference':float((native-original[:,0]).abs().max()),
        'author_forward_uses_rope':False,'preprocessing':'224 grayscale replicatedRGB,ImageNet mean/std; synthetic pixels only',
        'elapsed_seconds':time.monotonic()-before,'clinical_accuracy_measured':False,'gpu_runtime_measured':False,
        'module_level_ssl_override_executed':False}
    (research/'orthofoundation/compatibility_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
