"""Verify explicit-noise adapter against the original seeded CPU feature path."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))
from orthodiffusion_encoder import load_encoder, feature_with_noise, sha

def main():
    out = ROOT/'runs/orthodiffusion_resource_v1'
    out.mkdir(exist_ok=True)
    assert not (out/'cpu_fixture.npz').exists(), 'preserve existing fixture'
    src = ROOT/'research/knee_pretraining/orthodiffusion'
    receipt = json.loads((src/'source_receipt.json').read_text())
    for item in receipt['files']:
        assert sha(src/'source'/item['path']) == item['sha256']
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    ckpt = json.loads((src/'checkpoint_receipt.json').read_text())
    model = load_encoder(src/'source', ROOT/ckpt['file'], ckpt['sha256'])
    old = np.load(src/'cpu_fixture_64.npz')
    volume = torch.from_numpy(old['volume'])
    torch.manual_seed(20261016)
    noise = torch.randn_like(volume)
    with torch.inference_mode():
        actual = feature_with_noise(model, volume, noise)
        torch.manual_seed(20261016)
        original = model.get_feature(volume, torch.tensor([30]), name='mid_2', use_amp=False)
    error = float((actual-original).abs().max())
    assert error == 0
    assert np.array_equal(original.mean((2,3,4)).numpy(), old['pooled'])
    assert np.array_equal(original[...,::4,::4].numpy(), old['feature_probe'])
    np.savez_compressed(out/'cpu_fixture.npz', volume=volume.numpy(), noise=noise.numpy(),
                        feature=actual.numpy())
    result = dict(status='EXACT_AUTHOR_AND_PRIOR_FIXTURE_PARITY', max_abs_error=error,
                  shape=list(actual.shape), seed=20261016, timestep=30, block='mid_2',
                  adapter_sha256=sha(ROOT/'training/orthodiffusion_encoder.py'),
                  fixture_sha256=sha(out/'cpu_fixture.npz'),
                  original_fixture_sha256=sha(src/'cpu_fixture_64.npz'))
    (out/'cpu_fixture_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
