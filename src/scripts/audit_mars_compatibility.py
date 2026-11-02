"""Strict CPU encoder compatibility check; no task accuracy or GPU work."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
sys.path.insert(0, str(RUN / 'deps'))


def timeout(*_):
    raise TimeoutError('bounded CPU compatibility check expired')


def main():
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(150)
    import torch
    import monai
    import numpy
    from monai.networks.nets.swin_unetr import SwinTransformer
    torch.set_num_threads(2)
    assert monai.__version__ == '1.4.0'
    path = RUN / 'MARS.pt'
    receipt = json.loads((RUN / 'checkpoint_receipt.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert all(k.startswith('swinViT.') for k in state)
    state = {k.removeprefix('swinViT.'): v for k, v in state.items()}
    assert all(torch.isfinite(v).all().item() for v in state.values())
    # Exact dimensions from the pinned author's SwinVitWrapper and Model_Var.
    model = SwinTransformer(in_chans=1, embed_dim=48, window_size=(7, 7, 7),
                            patch_size=(2, 2, 2), depths=(2, 2, 2, 2),
                            num_heads=(3, 6, 12, 24), mlp_ratio=4., qkv_bias=True,
                            norm_layer=torch.nn.LayerNorm, use_checkpoint=False,
                            spatial_dims=3, use_v2=True).cpu().eval().requires_grad_(False)
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    x = torch.linspace(-1, 1, 96**3).reshape(1, 1, 96, 96, 96)
    started = time.monotonic()
    with torch.inference_mode():
        feature = model(x)[4]
        alternate = model(torch.zeros_like(x))[4]
    assert feature.shape == (1, 768, 3, 3, 3)
    assert torch.isfinite(feature).all() and torch.isfinite(alternate).all()
    difference = float((feature - alternate).abs().max())
    assert difference > 1e-6, 'Different synthetic images yielded identical features'
    result = {'status': 'PASS_STRICT_ENCODER_LOAD_AND_96_CUBE_CPU_FORWARD',
              'checkpoint_sha256': receipt['sha256'], 'monai_version': monai.__version__,
              'torch_version': torch.__version__, 'missing_keys': [], 'unexpected_keys': [],
              'numpy_version': numpy.__version__,
              'encoder_parameters': sum(p.numel() for p in model.parameters()),
              'feature_shape': list(feature.shape), 'different_input_max_abs_difference': difference,
              'two_forward_seconds_cpu': time.monotonic() - started,
              'task_accuracy_measured': False, 'gpu_used': False,
              'limitations': ['Synthetic tensor check only; DICOM preprocessing and task transfer remain untested.',
                             'Local CPU runtime is not a GPU cost estimate.',
                             'No independent upstream checkpoint checksum is published in the reviewed source.']}
    (RUN / 'compatibility_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    signal.alarm(0)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
