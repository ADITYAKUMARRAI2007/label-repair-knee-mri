"""Verify a private uploaded snapshot and retain its small evaluation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def download(api, dataset, name, directory):
    api.dataset_download_file(dataset, name, path=str(directory), force=True, quiet=True)
    path = directory / Path(name).name
    if not path.exists():
        archive = directory / (path.name + '.zip')
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() in ([name], [path.name]), 'Unexpected download archive contents'
            with z.open(z.namelist()[0]) as source, path.open('wb') as target:
                for block in iter(lambda: source.read(8 << 20), b''): target.write(block)
        archive.unlink()
    return path


def verify_expanded(api, dataset, stem, receipt, listing, directory, destination):
    prefix = receipt['prefix']; expected = receipt['members']
    allowed = {f'{prefix}_{s}' for s in ['last.pt','ema.pt','history.json','split.json',
                                        'manifest.json','effective_weights.json','oof_predictions.csv']}
    assert expected and set(expected) <= allowed and f'{prefix}_last.pt' in expected
    paths = {name: stem + '/' + name for name in expected}
    assert set(listing) == {stem + '.json', *paths.values()}, 'Unexpected expanded dataset layout'
    verified, small = {}, {}
    for name, remote in paths.items():
        assert listing[remote] == expected[name]['bytes']
        artifact = download(api, dataset, remote, directory)
        assert artifact.stat().st_size == expected[name]['bytes'] and sha(artifact) == expected[name]['sha256']
        verified[name] = expected[name]
        if not name.endswith('.pt'):
            assert artifact.stat().st_size < 2 << 20
            small[name] = artifact.read_bytes()
        artifact.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in small.items(): (destination / name).write_bytes(data)
    return verified


def verify_bundle(bundle, receipt, destination):
    assert bundle.stat().st_size == receipt['bundle_bytes']
    assert sha(bundle) == receipt['bundle_sha256'], 'Bundle digest mismatch'
    prefix = receipt['prefix']
    allowed = {f'{prefix}_{s}' for s in ['last.pt', 'ema.pt', 'history.json', 'split.json',
                                        'manifest.json', 'effective_weights.json', 'oof_predictions.csv']}
    expected = receipt['members']
    assert expected and set(expected) <= allowed and f'{prefix}_last.pt' in expected
    verified, small = {}, {}
    with tarfile.open(bundle, 'r:') as tar:
        for item in tar:
            assert item.isfile() and item.name in expected and item.name not in verified
            assert Path(item.name).name == item.name and item.size == expected[item.name]['bytes']
            h = hashlib.sha256(); count = 0
            keep = not item.name.endswith('.pt')
            if keep: assert item.size < 2 << 20
            parts = []
            with tar.extractfile(item) as f:
                for block in iter(lambda: f.read(8 << 20), b''):
                    h.update(block); count += len(block)
                    if keep: parts.append(block)
            assert count == item.size and h.hexdigest() == expected[item.name]['sha256']
            verified[item.name] = {'bytes': count, 'sha256': h.hexdigest()}
            if keep: small[item.name] = b''.join(parts)
    assert set(verified) == set(expected), 'Missing tar members'
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in small.items(): (destination / name).write_bytes(value)
    return verified


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--phase', choices=['e01', 'e05', 'e09', 'final', 'terminal'], required=True)
    a = p.parse_args(); root = ROOT / f'runs/e05_full_fold{a.fold}'
    state = json.loads((root / f'commit_{a.phase}.json').read_text())
    assert state['status'] == 'PRIVATE_READY' and state['private_verified']
    prefix = f'e05_uniform_fold{a.fold}'; name = f'{prefix}_{a.phase}'
    destination = root / f'verified_{a.phase}'
    api = KaggleApi(); api.authenticate()
    with tempfile.TemporaryDirectory(prefix='snapshot_', dir=root) as tmp:
        directory = Path(tmp)
        path = download(api, state['dataset'], name + '.json', directory)
        assert path.stat().st_size == 64 * 1024
        receipt = json.loads(path.read_text())
        if a.phase == 'terminal':
            destination.mkdir(exist_ok=True)
            (destination / 'terminal.json').write_bytes(path.read_bytes())
            print(json.dumps(receipt, indent=2)); return
        assert receipt['status'] == 'CHECKPOINT_SNAPSHOT_UPLOADED'
        assert receipt['prefix'] == prefix and receipt['phase'] == a.phase
        if a.phase != 'final': assert receipt['completed_epoch'] == int(a.phase[1:])
        assert 1 <= receipt['completed_epoch'] <= 14
        remote = api.dataset_list_files(state['dataset']).to_dict()
        assert not remote.get('nextPageToken'), 'Unexpected pagination for small snapshot'
        listing = {f['name']: f['totalBytes'] for f in remote['datasetFiles']}
        bundle_preserved = name + '.tar' in listing
        if bundle_preserved:
            bundle = download(api, state['dataset'], name + '.tar', directory)
            members = verify_bundle(bundle, receipt, destination)
        else:
            members = verify_expanded(api, state['dataset'], name, receipt, listing, directory, destination)
        (destination / 'snapshot_receipt.json').write_bytes(path.read_bytes())
        result = {'status': 'PASS_REMOTE_SNAPSHOT_BYTES', 'dataset': state['dataset'],
                  'phase': a.phase, 'completed_epoch': receipt['completed_epoch'],
                  'bundle_sha256': receipt['bundle_sha256'], 'members': members,
                  'bundle_digest_independently_verified': bundle_preserved,
                  'all_member_bytes_independently_verified': True,
                  'platform_expanded_archive': not bundle_preserved,
                  'checkpoint_tensors_deserialized': False, 'accuracy_audited': False,
                  'competition_submission_enabled': False}
        (destination / 'integrity.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'status': 'SNAPSHOT_AUDIT_FAILED', 'error_type': type(error).__name__}))
        raise SystemExit(1)
