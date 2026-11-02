"""Upload exactly the verified model-only bundle as a private Kaggle dataset."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt_path = ROOT / 'runs/orthofoundation_asset_upload.json'
    if receipt_path.exists():
        raise RuntimeError('upload already attempted; inspect original dataset before any retry')
    directory = ROOT / 'datasets/orthofoundation_assets_v1'
    package = json.loads((ROOT / 'runs/orthofoundation_asset_package.json').read_text())
    if package['status'] != 'READY_PRIVATE_UPLOAD':
        raise RuntimeError('package not ready')
    if {p.name for p in directory.iterdir()} != set(package['files']):
        raise RuntimeError('unexpected packaged file')
    for name, entry in package['files'].items():
        p = directory / name
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        if p.stat().st_size != entry['bytes'] or h.hexdigest() != entry['sha256']:
            raise RuntimeError('packaged bytes changed: ' + name)
    metadata = json.loads((directory / 'dataset-metadata.json').read_text())
    api = KaggleApi(); api.authenticate()
    # Kaggle status returns403 for some nonexistent private slugs. Establish
    # absence through the successfully authenticated owned-dataset listing.
    owned = set()
    for page in range(1, 101):
        rows = api.dataset_list(mine=True, page=page) or []
        if not rows:
            break
        owned.update(r.ref for r in rows if r and r.ref)
    else:
        raise RuntimeError('owned-dataset listing did not terminate')
    if 'adityakumarrai0401/rsna-knee-e04-convnext-imagenet-weights' not in owned:
        raise RuntimeError('known owned dataset missing from authenticated listing')
    if metadata['id'] in owned:
        raise RuntimeError('dataset already exists; observe it')
    receipt = {'status': 'UPLOAD_ATTEMPT_REQUIRES_OBSERVATION', 'dataset': metadata['id'],
               'private': True, 'observed_utc': datetime.now(timezone.utc).isoformat()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    result = api.dataset_create_new(str(directory), public=False, quiet=True, convert_to_csv=False)
    error = str(getattr(result, 'error', None) or '').strip()
    receipt.update(status='UPLOAD_REJECTED' if error else 'UPLOAD_ACCEPTED', error=error or None)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
