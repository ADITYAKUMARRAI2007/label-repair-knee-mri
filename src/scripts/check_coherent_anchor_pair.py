"""Verify package identities, paired selectors and the real metadata facade."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/window_integrity_20261010'
sys.path.insert(0, str(ROOT / 'inference'))
from coherent_anchor_pair import prepare_cohort, read_input_audit
from coherent_triplets import coherent_centers


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = ROOT / 'public_kernels/coherent_anchor_pair'
    code = package / 'pair.py'
    build = json.loads((RUN / 'pair_build.json').read_text())
    assert sha(code) == build['code_sha256']
    assert sha(package / 'kernel-metadata.json') == build['metadata_sha256']
    tree = ast.parse(code.read_text())
    modules = ast.literal_eval(tree.body[0].value)
    contract = json.loads(modules['pair_contract.json'])
    assert hashlib.sha256(modules['pair_contract.json'].encode()).hexdigest() == build['contract_sha256']
    for name, expected in contract['module_sha256'].items():
        assert hashlib.sha256(modules[name].encode()).hexdigest() == expected, name
    assert not contract['competition_submission_enabled'] and not contract['weight_search_allowed']
    functions = {}
    for arm in ['control', 'candidate']:
        scope = {'np': np, 'SLOT_SIZES': [26, 22, 18, 12, 18], 'coherent_centers': coherent_centers}
        nodes = {node.name: node for node in ast.parse(modules[f'{arm}.py']).body
                 if isinstance(node, ast.FunctionDef)}
        for name in ['_eval_centers', 'find_test_root']:
            exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), '<selector>', 'exec'), scope)
        assert scope['find_test_root']() == contract['cohort_path']
        functions[arm] = scope['_eval_centers']
    cases = []
    masks = [np.ones(96, dtype=np.uint8)]
    for lo, hi in [(26, 48), (0, 26), (78, 96), (48, 66)]:
        mask = masks[0].copy()
        mask[lo:hi] = 0
        masks.append(mask)
    for mask in masks:
        control = functions['control'](mask, 96, 94)
        candidate = functions['candidate'](mask, 96, 94)
        assert candidate == coherent_centers(control, mask, [26, 22, 18, 12, 18])
        cases.append({'present_slices': int(mask.sum()), 'control_windows': len(control),
                      'candidate_windows': len(candidate)})
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        data = base / 'data'
        data.mkdir()
        for name in ['train.csv', 'train_series.csv', 'sample_submission.csv']:
            shutil.copyfile(ROOT / 'data' / name, data / name)
        (data / 'train_series').mkdir()
        cohort = base / 'cohort'
        prepare_cohort(data, cohort, contract['studies'])
        ids = pd.read_csv(cohort / 'test.csv')['StudyInstanceUID'].tolist()
        assert ids == contract['studies']
        assert pd.read_csv(cohort / 'test.csv').columns.tolist() == ['StudyInstanceUID']
        sample = pd.read_csv(cohort / 'sample_submission.csv')
        assert sample.shape == (78, 13) and (sample.iloc[:, 1:].to_numpy() == .5).all()
        assert (cohort / 'test_series').resolve() == (data / 'train_series').resolve()
        assert not (cohort / 'train.csv').exists()
        log = base / 'observations.log'
        log.write_text('\n'.join('PAIRED_INPUT_AUDIT ' + json.dumps({'study': uid}) for uid in ids))
        assert set(read_input_audit(log, ids)) == set(ids)
        for invalid in [ids[:-1], ids[:-1] + [ids[0]]]:
            log.write_text('\n'.join('PAIRED_INPUT_AUDIT ' + json.dumps({'study': uid}) for uid in invalid))
            try:
                read_input_audit(log, ids)
            except AssertionError:
                pass
            else:
                raise AssertionError('Incomplete or duplicate per-study observation accepted')
    # Freeze the future accuracy comparison before this candidate has predictions.
    archive = ROOT / 'runs/colab_n03/output'
    receipt = json.loads((archive / 'diagnostic_receipt.json').read_text())
    base_receipt = json.loads((archive / 'fixed_reader_blend/receipt.json').read_text())
    references = {'reader': archive / 'fixed_reader_blend/reader.csv',
                  'crop224': archive / 'n03_diagnostic.csv',
                  'old_anchor': archive / 'fixed_reader_blend/anchor.csv'}
    assert sha(references['reader']) == base_receipt['components']['reader.csv']
    assert sha(references['old_anchor']) == base_receipt['components']['anchor.csv']
    assert sha(references['crop224']) == receipt['files']['n03_diagnostic.csv']
    for path in references.values():
        assert pd.read_csv(path)['StudyInstanceUID'].tolist() == contract['studies']
    published = json.loads((ROOT / 'runs/tri_reader_20261010/protocol.json').read_text())['recipe']
    analysis = {'schema': 'coherent_anchor_fixed_comparison_v1',
                'pair_contract_sha256': build['contract_sha256'],
                'fixed_reference_sha256': {name: sha(path) for name, path in references.items()},
                'baseline_reader_weights': base_receipt['contract']['weights'],
                'tri_reader_recipe': published,
                'ranking': 'Average ties across the entire same78-study cohort; score the same58expert rows afterwards',
                'comparison': 'Candidate vs freshly reexecuted control for anchor, N02blend and N04blend; disclose control drift from archived anchor',
                'bootstrap': '2000paired report-group draws;seed20261010; fixed comparisons without weight search',
                'promotion': 'No automatic promotion or submission; evaluate paired uncertainty and per-target regressions',
                'accuracy_available': False, 'limitations': contract['limitations']}
    path = RUN / 'pair_analysis_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == analysis, 'Do not replace a frozen comparison'
    else:
        path.write_text(json.dumps(analysis, indent=2) + '\n')
    report = {'status': 'PASS_PAIRED_INPUT_PACKAGE_LOCAL_GATES', 'code_sha256': sha(code),
              'selector_cases': cases, 'real_cohort_metadata_rows': 78,
              'input_audit_missing_and_duplicate_rejection': True,
              'reference_prediction_identities_verified': True,
              'analysis_protocol_sha256': sha(path),
              'not_tested': ['CUDA execution', 'actual paired decoded images', 'candidate task accuracy'],
              'new_gpu_jobs': 0, 'new_competition_submissions': 0}
    (RUN / 'pair_local_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
