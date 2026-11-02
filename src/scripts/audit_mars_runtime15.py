"""Check a newer MONAI runtime against saved real-study encoder features."""
import hashlib
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
# Keep NumPy isolated as in the reference; replace only MONAI for this check.
sys.path[:0] = [str(RUN / 'runtime15/deps'), str(RUN / 'deps'),
               str(ROOT / 'inference')]


def timeout(*_):
    raise TimeoutError('Runtime comparison exceeded its CPU budget')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-merging', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(150)
    import monai
    import numpy as np
    import torch
    from monai.networks.nets.swin_unetr import SwinTransformer
    from mars_encoder import CHECKPOINT_SHA256, encode_study_sequences, load_encoder
    from mars_legacy_merging import MarsLegacyPatchMerging

    torch.set_num_threads(2)
    assert monai.__version__ == '1.5.1'
    download = json.loads((RUN / 'runtime15/download_receipt.json').read_text())
    assert sha(RUN / 'runtime15' / download['filename']) == download['sha256']
    checkpoint = RUN / 'MARS.pt'
    assert sha(checkpoint) == CHECKPOINT_SHA256
    fixture_path = RUN / 'real_study_cpu_fixture.npz'
    with np.load(fixture_path, allow_pickle=False) as fixture:
        volumes = fixture['volumes']
        present = fixture['present']
        expected = fixture['features']
    if args.legacy_merging:
        model = load_encoder(checkpoint)
    else:
        model = SwinTransformer(in_chans=1, embed_dim=48, window_size=(7, 7, 7),
                                patch_size=(2, 2, 2), depths=(2, 2, 2, 2),
                                num_heads=(3, 6, 12, 24), mlp_ratio=4., qkv_bias=True,
                                norm_layer=torch.nn.LayerNorm, use_checkpoint=False,
                                spatial_dims=3, use_v2=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        assert all(k.startswith('swinViT.') for k in state)
        loaded = model.load_state_dict({k.removeprefix('swinViT.'): v for k, v in state.items()}, strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
    model.eval().requires_grad_(False)
    started = time.monotonic()
    actual = encode_study_sequences(model, torch.from_numpy(volumes[None]),
                                    torch.from_numpy(present[None])).numpy()
    max_error = float(np.max(np.abs(actual - expected)))
    finite = bool(np.isfinite(actual).all())
    passed = finite and max_error <= 1e-6
    result = {
        'status': ('PASS_REAL_STUDY_MONAI15_REFERENCE_FEATURE_PARITY_CPU' if passed
                   else 'FAIL_REAL_STUDY_MONAI15_REFERENCE_FEATURE_PARITY_CPU'),
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'fixture_sha256': sha(fixture_path), 'monai_version': monai.__version__,
        'numpy_version': np.__version__, 'torch_version': torch.__version__,
        'wheel_sha256': download['sha256'], 'feature_shape': list(actual.shape),
        'legacy_merging_enabled': args.legacy_merging,
        'legacy_merging_sha256': sha(ROOT / 'inference/mars_legacy_merging.py'),
        'encoder_source_sha256': sha(ROOT / 'inference/mars_encoder.py'),
        'reference_feature_max_abs_error': max_error,
        'reference_feature_mean_abs_error': float(np.mean(np.abs(actual - expected))),
        'features_finite': finite,
        'forward_seconds_cpu': time.monotonic() - started,
        'gpu_used': False, 'task_accuracy_measured': False,
        'existing_adapter_modified': args.legacy_merging,
        'limitations': [
            'One real study and CPU runtime only; Kaggle CUDA/Python3.13 parity remains untested.',
            'NumPy1.26 used here; support for newer NumPy is from package requirements, not execution.',
            'Compatibility does not establish clinical transfer or an expected score.'],
    }
    name = 'legacy_compatibility_audit.json' if args.legacy_merging else 'compatibility_audit.json'
    (RUN / 'runtime15' / name).write_text(json.dumps(result, indent=2) + '\n')
    signal.alarm(0)
    print(json.dumps(result))
    if not passed:
        raise RuntimeError('New runtime changes reference features; keep the original adapter')


if __name__ == '__main__':
    main()
