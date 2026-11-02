"""Reconstruct frozen training inputs from native competition/public CSV mounts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from build_e04_pilot import extract

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'

DRIVER = '''
def reconstruct(workspace, train_path, inputs):
    workspace, train_path, inputs = Path(workspace), Path(train_path), Path(inputs)
    def find_unique(name, expected):
        receipt = METADATA['fold_reconstruction_only_sources'][name]
        parent = Path(receipt['path']).parent.name
        owner, slug = parent.split('_', 1)
        candidates = [inputs / receipt['path'], inputs / slug / name,
                      inputs / 'datasets' / owner / slug / name]
        valid = [p for p in candidates if p.is_file() and sha256_file(p) == expected]
        if not valid: raise FileNotFoundError('Required public label CSV: ' + name)
        return valid[0]
    # The historical reconstruction uses a supplied find_unique helper.
    globals()['find_unique'] = find_unique
    train = pd.read_csv(train_path, dtype={'StudyInstanceUID': str})
    ids = np.asarray(sorted(train.StudyInstanceUID.astype(str)), dtype=str)
    y, weights, gold, folds, groups, audit = reconstruct_frozen_inputs(train_path, ids)
    assert audit == METADATA['split_audit']
    reconstructed = dict(METADATA)
    reconstructed['rows'] = [{'StudyInstanceUID': str(uid), 'fold': int(f),
                              'report_group': int(g), 'expert': bool(expert)}
                             for uid, f, g, expert in zip(ids, folds, groups, gold)]
    output = workspace / 'runs/E04_contract'; output.mkdir(parents=True, exist_ok=False)
    metadata = output / 'e04_contract.json'
    metadata.write_text(json.dumps(reconstructed, indent=2) + '\\n')
    np.save(output / 'e04_targets.npy', y)
    np.save(output / 'e04_weights.npy', weights)
    assert sha256_file(metadata) == EXPECTED_METADATA_SHA256
    for name, expected in reconstructed['arrays'].items():
        assert sha256_file(output / name) == expected, name
    return {'status': 'EXACT_FROZEN_INPUTS_RECONSTRUCTED_FROM_NATIVE_PUBLIC_SOURCES',
            'studies': len(ids), 'experts': int(gold.sum()),
            'metadata_sha256': sha256_file(metadata),
            'local_image_feature_or_label_array_upload_needed': False}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(reconstruct(args.workspace, args.train, args.inputs)))
'''


def main():
    path = ROOT / 'runs/E04_contract/e04_contract.json'
    frozen = json.loads(path.read_text())
    metadata = dict(frozen); metadata['rows'] = None
    algorithm = extract(ROOT / 'scripts/t03_contract.py',
                        {'TARGETS', 'PUBLIC_LABEL_SHA256', '_normalise_report', '_simhash',
                         'report_groups', 'make_folds', 'sha256_file'})
    constants = '\n'.join([
        f"E04_TRAIN_SHA256 = {frozen['official_train_sha256']!r}",
        f"E04_FOLD_SHA256 = {frozen['split_audit']['fold_sha256']!r}",
        f"E04_TARGET_SHA256 = {frozen['arrays']['e04_targets.npy']!r}",
        f"E04_WEIGHT_SHA256 = {frozen['arrays']['e04_weights.npy']!r}"])
    recipe = (ROOT / 'training/e04_input_contract.py').read_text().replace(
        '# __FROZEN_HASHES__', constants).replace('# __SPLIT_ALGORITHM__', algorithm)
    source = ('import hashlib,json,re,unicodedata\nfrom pathlib import Path\n'
              'import numpy as np\nimport pandas as pd\n' + recipe + '\n' +
              'METADATA=' + repr(metadata) + '\n' +
              'EXPECTED_METADATA_SHA256=' + repr(hashlib.sha256(path.read_bytes()).hexdigest()) + '\n' + DRIVER)
    compile(source, 'mars_runtime_inputs.py', 'exec')
    destination = ROOT / 'training/mars_runtime_inputs.py'
    destination.write_text(source)
    record = {'status': 'NATIVE_INPUT_RECONSTRUCTION_PREPARED',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'competition_records_embedded': False, 'local_data_array_upload_needed': False,
              'runtime_reconstruction_verified': False}
    (BASE / 'runtime_input_preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
