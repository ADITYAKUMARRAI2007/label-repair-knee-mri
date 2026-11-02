"""Check acquisition and mask invariants against the actual public selector."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from coherent_triplets import coherent_centers


def main():
    path = ROOT / 'public_kernels/published_tri_reader/rsna-knee-n04-published-tri-reader.py'
    tree = ast.parse(path.read_text())
    modules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(x, ast.Name) and x.id == 'MODULES' for x in n.targets))
    source = modules['anchor/published_inference.py']
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_eval_centers')
    scope = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<published_selector>', 'exec'), scope)
    sizes = [26, 22, 18, 12, 18]
    slot_ids = np.repeat(np.arange(5), sizes)
    mask = np.ones(96, dtype=np.uint8)
    original = scope['_eval_centers'](mask, 96, 94)
    filtered = coherent_centers(original, mask, sizes)
    assert len(original) == 94 and len(filtered) == 86
    assert sorted(set(original)-set(filtered)) == [25, 26, 47, 48, 65, 66, 77, 78]
    # Values encode acquisition identity; every retained triplet has one identity.
    tagged_volume = np.broadcast_to(slot_ids[:, None, None], (96, 2, 2))
    for center in filtered:
        assert len(np.unique(tagged_volume[center-1:center+2])) == 1
    gap_mask = mask.copy()
    gap_mask[26:48] = 0
    selected = scope['_eval_centers'](gap_mask, 96, 94)
    actual = coherent_centers(selected, gap_mask, sizes)
    assert all(gap_mask[c-1:c+2].all() for c in actual)
    assert actual == [c for c in selected if c in filtered and gap_mask[c-1:c+2].all()]
    repeated = coherent_centers([1, 1, 2, 25, 26, 27], mask, sizes)
    assert repeated == [1, 1, 2, 27], 'Original order and multiplicity changed'
    for selected, bad_mask in [([1], np.zeros(96)), ([0], mask), ([95], mask), ([1], mask[:95])]:
        try:
            coherent_centers(selected, bad_mask, sizes)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid or empty triplets accepted')
    record = {'status': 'PASS_ACQUISITION_AND_MASK_INVARIANTS',
              'original_windows': len(original), 'retained_full_mask_windows': len(filtered),
              'retained_internal_missing_slot_windows': len(actual),
              'source_package_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'helper_sha256': hashlib.sha256((ROOT / 'inference/coherent_triplets.py').read_bytes()).hexdigest(),
              'checks': ['Actual published selector replay', 'Single-acquisition window identity',
                         'No missing-slot pixels selected', 'Preserved ordering and repetition',
                         'Empty and out-of-range windows rejected'],
              'gpu_launched': False, 'accuracy_tested': False,
              'limitation': 'Source/input checks only. Frozen models trained with the original windows may lose accuracy after filtering.'}
    out = ROOT / 'runs/window_integrity_20261010'
    out.mkdir(exist_ok=True)
    (out / 'local_check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
