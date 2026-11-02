"""Create a private-upload-ready, pinned offline package and CPU fixture."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import torch
    base = ROOT / 'research/knee_pretraining'
    out = ROOT / 'datasets/orthofoundation_assets_v1'
    out.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'runs/orthofoundation_asset_upload.json').exists():
        raise RuntimeError('preserve uploaded assets')
    source = base / 'dinov3/source'
    receipts = {name: json.loads((base / name / 'source_receipt.json').read_text())
                for name in ['dinov3', 'orthofoundation']}
    for name, receipt in receipts.items():
        for entry in receipt['files']:
            if sha(base / name / 'source' / entry['path']) != entry['sha256']:
                raise RuntimeError('pinned source differs')
    checkpoint = ROOT / 'models/orthofoundation/OrthoFoudation-L.pth'
    expected = json.loads((base / 'orthofoundation/checkpoint_receipt.json').read_text())
    if sha(checkpoint) != expected['sha256']:
        raise RuntimeError('checkpoint hash differs')
    target = out / checkpoint.name
    if not target.exists():
        os.link(checkpoint, target)
    if sha(target) != expected['sha256']:
        raise RuntimeError('packaged weights differ')
    (out / 'dinov3_source.json').write_text(json.dumps({
        entry['path']: (source / entry['path']).read_text()
        for entry in receipts['dinov3']['files']}, sort_keys=True) + '\n')
    (out / 'dinov3_source.zip').unlink(missing_ok=True)
    (out / 'OrthoFoundation_LICENSE.txt').write_bytes((base / 'orthofoundation/source/LICENSE').read_bytes())
    (out / 'DINOv3_LICENSE.md').write_bytes((source / 'LICENSE.md').read_bytes())
    sys.path[:0] = [str(ROOT / 'training'), str(source)]
    from orthofoundation_encoder import OrthoEncoder, normalize_pixels
    torch.set_num_threads(2)
    model = OrthoEncoder(checkpoint)
    gray = torch.linspace(0, 255, 224 * 224).reshape(1, 1, 224, 224)
    x = normalize_pixels(gray)
    with torch.inference_mode():
        actual = model(x)
        tokens, _ = model.backbone.prepare_tokens_with_masks(x)
        for block in model.backbone.blocks:
            tokens = block(tokens)
        tokens = model.backbone.norm(tokens)
        reference = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], 1)
    if not torch.equal(actual, reference) or not torch.isfinite(actual).all():
        raise RuntimeError('adapter pooling parity failed')
    np.savez(out / 'orthofoundation_cpu_fixture.npz', pixels=gray.numpy(), features=actual.numpy())
    provenance = {'status': 'PINNED_CPU_ADAPTER_VERIFIED', 'sources': receipts,
                  'checkpoint': expected, 'adapter_sha256': sha(ROOT / 'training/orthofoundation_encoder.py'),
                  'forward': 'author full-finetune wrapper; no RoPE',
                  'pooling': 'CLS concatenated with mean of all 196 patches; excludes four registers',
                  'cpu_pooling_max_abs_error': float((actual - reference).abs().max()),
                  'accuracy_measured': False, 'upstream_competition_cohort_overlap': 'unknown'}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'README.md').write_text('''# OrthoFoundation offline experiment assets

Private working package. Built with DINOv3.
OrthoFoundation: https://github.com/ytrsk/OrthoFoundation
DINOv3: https://github.com/facebookresearch/dinov3

Retain both accompanying licenses. The author's repository has an MIT license;
the upstream DINOv3 architecture and pretrained derivatives are subject to the
included DINOv3 License. This package does not relicense those materials.
See provenance.json for source commits, file hashes and checkpoint origin.
The CPU fixture tests numerical compatibility only, not diagnostic accuracy.
No competition images, labels, account credentials or private reports included.
''')
    (out / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'RSNA Knee OrthoFoundation Assets V1',
        'id': 'adityakumarrai0401/rsna-knee-orthofoundation-assets-v1',
        'licenses': [{'name': 'other'}],
        'description': 'Private experiment assets. Built with DINOv3. See included licenses and provenance.'
    }, indent=2) + '\n')
    manifest = {'status': 'READY_PRIVATE_UPLOAD', 'files': {
        p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.iterdir()) if p.is_file()}}
    (ROOT / 'runs/orthofoundation_asset_package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'file_count': len(manifest['files']),
                      'cpu_adapter_max_abs_error': 0, 'bytes': sum(r['bytes'] for r in manifest['files'].values())}))


if __name__ == '__main__':
    main()
