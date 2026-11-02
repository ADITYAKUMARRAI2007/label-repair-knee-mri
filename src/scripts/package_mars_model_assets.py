"""Package only publicly released weights, dependencies and licensing."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
OUT = ROOT / 'datasets/mars_model_assets_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (RUN / 'model_asset_upload.json').exists()
    download = json.loads((RUN / 'runtime15/download_receipt.json').read_text())
    checkpoint = json.loads((RUN / 'checkpoint_receipt.json').read_text())
    license_receipt = json.loads((RUN / 'license_receipt.json').read_text())
    payload = {
        'MARS.pt': (RUN / 'MARS.pt', checkpoint['sha256']),
        download['filename']: (RUN / 'runtime15' / download['filename'], download['sha256']),
        'MARS_LICENSE.txt': (RUN / 'MARS_LICENSE.txt', license_receipt['sha256']),
    }
    OUT.mkdir(exist_ok=True)
    for name, (source, expected) in payload.items():
        assert sha(source) == expected, name
        target = OUT / name
        if not target.exists():
            os.link(source, target)
        assert sha(target) == expected, name
    provenance = {
        'checkpoint': checkpoint, 'repository': json.loads((RUN / 'source_receipt.json').read_text()),
        'repository_license': license_receipt, 'monai_distribution': download,
        'private_images_included': False, 'patient_feature_arrays_included': False,
        'competition_labels_or_reports_included': False,
        'public_sources_only': True,
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUT / 'NOTICE.md').write_text('''# Public-source MARS model assets

MARS: https://github.com/zqiuak/MARS
MONAI: https://github.com/Project-MONAI/MONAI

This private working package contains only the public author-linked MARS pretrained
checkpoint, the official MONAI1.5.1 distribution, licenses and public provenance.
It contains no MRI test fixtures, patient feature arrays, competition labels or
private reports. Numerical references will be built inside the competition mount.

Keep the accompanying Apache2.0 repository license and the MONAI license inside
its wheel. This package does not relicense model weights or other source material.
See provenance.json for public sources and hashes. No accuracy claim is made.
''')
    (OUT / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'RSNA Knee MARS Model Assets V2',
        'id': 'adityakumarrai0401/rsna-knee-mars-model-assets-v2',
        'licenses': [{'name': 'other'}],
        'description': 'Public pretrained model, official dependencies and licenses only. No MRI fixtures or patient features.',
    }, indent=2) + '\n')
    expected_names = set(payload) | {'provenance.json', 'NOTICE.md', 'dataset-metadata.json'}
    assert {p.name for p in OUT.iterdir()} == expected_names
    manifest = {
        'status': 'READY_PRIVATE_PUBLIC_MODEL_ONLY_UPLOAD', 'private': True,
        'public_sources_only': True, 'private_images_or_patient_features_included': False,
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.iterdir())},
        'new_gpu_jobs': 0, 'new_competition_submissions': 0,
    }
    (RUN / 'model_asset_package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'files': list(manifest['files']),
                      'bytes': sum(r['bytes'] for r in manifest['files'].values())}))


if __name__ == '__main__':
    main()
