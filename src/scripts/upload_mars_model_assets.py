"""Upload public model assets only, following the rejected MRI-fixture proposal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expired(*_):
    raise TimeoutError('Observe the same attempt without repeating the upload')


def main():
    receipt_path = RUN / 'model_asset_upload.json'
    assert not receipt_path.exists()
    directory = ROOT / 'datasets/mars_model_assets_v2'
    manifest_path = RUN / 'model_asset_package.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'READY_PRIVATE_PUBLIC_MODEL_ONLY_UPLOAD'
    assert manifest['private'] and manifest['public_sources_only']
    assert manifest['private_images_or_patient_features_included'] is False
    expected = {'MARS.pt', 'monai-1.5.1-py3-none-any.whl', 'MARS_LICENSE.txt',
                'provenance.json', 'NOTICE.md', 'dataset-metadata.json'}
    assert {p.name for p in directory.iterdir()} == set(manifest['files']) == expected
    for name, record in manifest['files'].items():
        path = directory / name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    metadata = json.loads((directory / 'dataset-metadata.json').read_text())
    assert metadata['id'] == 'adityakumarrai0401/rsna-knee-mars-model-assets-v2'
    signal.signal(signal.SIGALRM, expired)
    record = None
    try:
        api = KaggleApi(); api.authenticate()
        signal.alarm(180)
        owned = set()
        for page in range(1, 101):
            rows = api.dataset_list(mine=True, page=page) or []
            if not rows:
                break
            owned.update(row.ref for row in rows if row and row.ref)
        else:
            raise AssertionError('Owned dataset pagination did not finish')
        assert 'adityakumarrai0401/rsna-knee-decoder-cp313' in owned
        assert 'adityakumarrai0401/rsna-knee-e04-convnext-imagenet-weights' in owned
        assert metadata['id'] not in owned
        record = {
            'status': 'UPLOAD_ATTEMPT_REQUIRES_OBSERVATION', 'dataset': metadata['id'],
            'private': True, 'public_sources_only': True, 'private_images_or_patient_features_included': False,
            'known_account_datasets_verified': True, 'asset_package_sha256': sha(manifest_path),
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'new_gpu_jobs': 0, 'new_competition_submissions': 0,
        }
        with receipt_path.open('x') as handle:
            handle.write(json.dumps(record, indent=2) + '\n')
        response = api.dataset_create_new(str(directory), public=False, quiet=True, convert_to_csv=False)
        error_present = bool(getattr(response, 'error', None))
        record.update(status='UPLOAD_REJECTED' if error_present else 'UPLOAD_ACCEPTED',
                      error_present=error_present, returned_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        if record is None:
            print(json.dumps({'status': 'PREFLIGHT_FAILED_NO_UPLOAD', 'error_type': type(error).__name__}))
            return
        record['observation_error_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        if record is not None:
            receipt_path.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record))


if __name__ == '__main__':
    main()
