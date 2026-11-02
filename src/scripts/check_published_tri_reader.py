"""Check packaged source identity, failure paths and exact saved-prediction fusion."""
import ast
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/published_tri_reader'
PACKAGE = ROOT / 'public_kernels/published_tri_reader'
sys.path.insert(0, str(ROOT / 'inference'))
from fixed_reader_blend_v3 import blend


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = json.loads((RUN / 'build.json').read_text())
    meta = json.loads((PACKAGE / 'kernel-metadata.json').read_text())
    path = PACKAGE / meta['code_file']
    assert sha(path) == build['code_sha256']
    assert sha(PACKAGE / 'kernel-metadata.json') == build['metadata_sha256']
    tree = ast.parse(path.read_text())
    modules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MODULES' for t in n.targets))
    contract = json.loads(modules['tri_reader_contract.json'])
    assert modules['tri_reader_contract.json'] == (RUN / 'contract.json').read_text()
    for name, expected in contract['module_sha256'].items():
        assert hashlib.sha256(modules[name].encode()).hexdigest() == expected, name
    assert meta['id'] == build['kernel'] and meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['title'].lower().replace(' ', '-') == meta['id'].split('/')[1]
    crop = ast.parse(modules['crop224.py'])
    functions = {n.name: n for n in crop.body if isinstance(n, ast.FunctionDef)}

    def extract(node, scope):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<guard-check>', 'exec'), scope)
        return scope[node.name]

    scope = {'torch': SimpleNamespace(load=lambda *a, **kw: (_ for _ in ()).throw(AssertionError('deserialized wrong bytes')))}
    loader = extract(functions['load_model'], scope)
    with tempfile.TemporaryDirectory() as temp:
        wrong = Path(temp) / 'wrong.pt'
        wrong.write_bytes(b'wrong checkpoint')
        try:
            loader(wrong, 'arch', 224, 'cpu')
        except RuntimeError as e:
            assert 'checkpoint bytes differ' in str(e)
        else:
            raise AssertionError('Invalid checkpoint accepted')
        scope = {'np': np, 'MAXS': 3, 'IMG': 2, '_SER': {}, '_TSDIR': '',
                 '_READER': lambda *a: (np.zeros((3, 2, 2), np.uint8), np.zeros(3, np.uint8))}
        pre = extract(functions['_pre_one'], scope)
        try:
            pre('empty')
        except RuntimeError as e:
            assert 'No decoded view' in str(e)
        else:
            raise AssertionError('Empty study accepted')
        base = ROOT / 'runs/colab_n03/output'
        sample = pd.read_csv(base / 'n02_diagnostic.csv', dtype={'StudyInstanceUID': str})
        test = Path(temp) / 'test.csv'
        sample[['StudyInstanceUID']].to_csv(test, index=False)
        sample_path = Path(temp) / 'sample_submission.csv'
        sample.to_csv(sample_path, index=False)
        spatial = Path(temp) / 'spatial.csv'
        final = Path(temp) / 'final.csv'
        blend(base / 'fixed_reader_blend/anchor.csv', base / 'n03_diagnostic.csv', spatial,
              {t: contract['crop_weight'] for t in contract['reader_weights']}, test, sample_path)
        blend(spatial, base / 'fixed_reader_blend/reader.csv', final, contract['reader_weights'], test, sample_path)
        actual = pd.read_csv(final)
        expected = pd.read_csv(ROOT / 'runs/tri_reader_20261010/diagnostic_predictions.csv')
        assert actual.StudyInstanceUID.tolist() == expected.StudyInstanceUID.tolist()
        labels = list(contract['reader_weights'])
        np.testing.assert_allclose(actual[labels], expected[labels], atol=1e-12, rtol=0)
    report = {'status': 'PASS_LOCAL_PACKAGE_AND_FUSION_CHECKS', 'code_sha256': sha(path),
              'contract_sha256': sha(RUN / 'contract.json'),
              'checks': ['All embedded source hashes', 'Title and slug match',
                         'Wrong checkpoint rejected before deserialization', 'Empty study rejected',
                         'CSV fusion matches exact user-linked recipe on78 saved studies'],
              'not_tested': ['Full three-reader CUDA execution', 'Hidden-test runtime', 'Official candidate score'],
              'gpu_launched': False, 'competition_submission_enabled': False}
    (RUN / 'local_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
