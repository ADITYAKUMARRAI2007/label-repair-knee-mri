"""Prepare private offline MARS assets; no upload or GPU allocation."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
OUT = ROOT / 'datasets/mars_assets_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (RUN / 'asset_upload.json').exists(), 'Preserve any attempted upload'
    protocol = json.loads((RUN / 'resource_protocol.json').read_text())
    audit = json.loads((RUN / 'runtime15/legacy_compatibility_audit.json').read_text())
    assert audit['status'] == 'PASS_REAL_STUDY_MONAI15_REFERENCE_FEATURE_PARITY_CPU'
    assert sha(ROOT / 'inference/mars_encoder.py') == audit['encoder_source_sha256']
    assert sha(ROOT / 'inference/mars_legacy_merging.py') == audit['legacy_merging_sha256']
    download = json.loads((RUN / 'runtime15/download_receipt.json').read_text())
    license_receipt = json.loads((RUN / 'license_receipt.json').read_text())
    files = {
        'MARS.pt': (RUN / 'MARS.pt', protocol['checkpoint_sha256']),
        'real_study_cpu_fixture.npz': (RUN / 'real_study_cpu_fixture.npz', protocol['fixture_sha256']),
        'real_study_input_receipt.json': (RUN / 'real_study_input_receipt.json',
                                        sha(RUN / 'real_study_input_receipt.json')),
        download['filename']: (RUN / 'runtime15' / download['filename'], protocol['monai_wheel_sha256']),
        'MARS_LICENSE.txt': (RUN / 'MARS_LICENSE.txt', license_receipt['sha256']),
    }
    OUT.mkdir(exist_ok=True)
    for name, (source, expected) in files.items():
        assert sha(source) == expected, name
        target = OUT / name
        if not target.exists():
            os.link(source, target)
        assert sha(target) == expected, name
    provenance = {
        'checkpoint': json.loads((RUN / 'checkpoint_receipt.json').read_text()),
        'repository': json.loads((RUN / 'source_receipt.json').read_text()),
        'repository_license': license_receipt, 'monai_download': download,
        'real_study_reference_audit': audit,
        'private_competition_fixture': True,
        'code_license_does_not_assert_additional_checkpoint_or_competition_data_rights': True,
        'resource_protocol_sha256': sha(RUN / 'resource_protocol.json'),
        'label_values_included': False, 'accuracy_measured': False,
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUT / 'NOTICE.md').write_text('''# Private MARS experiment assets

MARS: https://github.com/zqiuak/MARS
MONAI: https://github.com/Project-MONAI/MONAI

This private working package contains the publicly author-linked MARS checkpoint,
the official MONAI1.5.1 wheel, and one canonicalized competition study used as a
numerical reference. Keep the competition fixture private. It contains no target
values, but it does contain derived MRI image tensors.

Retain the included MARS license and the MONAI license inside its wheel.
Repository code is Apache2.0; this package does not relicense checkpoint weights
or competition data. See provenance.json for exact sources and hashes.

MONAI1.5's default patch merging changes this checkpoint's output. The inference
adapter preserves MONAI1.4 offsets; reference parity is verified on CPU only.
No task accuracy, CUDA resource budget or competition score is established here.
''')
    (OUT / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'RSNA Knee MARS Assets V1',
        'id': 'adityakumarrai0401/rsna-knee-mars-assets-v1',
        'licenses': [{'name': 'other'}],
        'description': 'Private model assets and private numerical MRI fixture. Retain original licenses and competition data terms.',
    }, indent=2) + '\n')
    expected = set(files) | {'provenance.json', 'NOTICE.md', 'dataset-metadata.json'}
    assert {p.name for p in OUT.iterdir()} == expected
    manifest = {
        'status': 'READY_PRIVATE_OFFLINE_RESOURCE_PILOT_ASSETS_NOT_UPLOADED',
        'private': True, 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                                 for p in sorted(OUT.iterdir())},
        'new_gpu_jobs': 0, 'new_competition_submissions': 0,
    }
    (RUN / 'asset_package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'files': len(manifest['files']),
                      'bytes': sum(r['bytes'] for r in manifest['files'].values())}))


if __name__ == '__main__':
    main()
