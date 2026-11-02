"""Independently replay executed CSV fusion; visible execution is not accuracy."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/published_tri_reader'
PACKAGE = ROOT / 'public_kernels/published_tri_reader'
sys.path.insert(0, str(ROOT / 'inference'))
from nartaa_reproduction_guard import sha, validate_submission, reported_failure


def audit(output):
    build = json.loads((RUN / 'build.json').read_text())
    meta = json.loads((PACKAGE / 'kernel-metadata.json').read_text())
    code = PACKAGE / meta['code_file']
    assert sha(code) == build['code_sha256']
    assert sha(PACKAGE / 'kernel-metadata.json') == build['metadata_sha256']
    assert sha(RUN / 'contract.json') == build['contract_sha256']
    tree = ast.parse(code.read_text())
    modules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MODULES' for t in n.targets))
    contract = json.loads(modules['tri_reader_contract.json'])
    root = output / 'tri_reader_source'
    folder = output / 'published_tri_reader'
    baseline_folder = output / 'fixed_reader_blend'
    for failure in [folder / 'failure.json', baseline_folder / 'failure.json', folder / 'rejected_submission.csv']:
        assert not failure.exists(), 'Failure artifact found: ' + failure.name
    receipt_path = folder / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'PASS_EXECUTION_ONLY' and receipt['contract'] == contract
    assert not receipt['accuracy_measured'] and not receipt['competition_submission_enabled']
    assert sha(root / 'tri_reader_contract.json') == build['contract_sha256']
    for name, expected in contract['module_sha256'].items():
        assert sha(root / name) == expected, name
    baseline = json.loads((baseline_folder / 'receipt.json').read_text())
    assert baseline['status'] == 'PASS_EXECUTION_ONLY'
    base_contract = json.loads(modules['contract.json'])
    assert baseline['contract'] == base_contract
    assert not baseline['accuracy_measured'] and not baseline['competition_submission_enabled']
    assert receipt['baseline_receipt_sha256'] == sha(baseline_folder / 'receipt.json')
    expected_anchor = json.loads(modules['anchor/contract.json'])
    assert baseline['anchor_receipt']['contract'] == expected_anchor
    assert baseline['anchor_receipt']['checkpoint_sha256'] == expected_anchor['checkpoint_sha256']
    assert baseline['anchor_receipt']['source_sha256'] == expected_anchor['source_sha256']
    assert all(baseline[name]['reported_failure_count'] == 0 for name in ['anchor', 'reader', 'decoder_check'])
    assert receipt['crop_execution']['reported_failure_count'] == 0
    assert 'DECODER_ABI_PASS cp313' in (baseline_folder / 'decoder.log').read_text()
    crop_log = (folder / 'crop224.log').read_text()
    assert 'N03_CHECKPOINT_SHA256 ' + contract['crop_checkpoint_sha256'] in crop_log
    for log_path in [baseline_folder / 'anchor.log', baseline_folder / 'reader.log', folder / 'crop224.log']:
        assert not any(reported_failure(line) for line in log_path.read_text().splitlines()), log_path.name
    test = ROOT / 'data/test.csv'
    sample = ROOT / 'data/sample_submission.csv'
    paths = {'anchor': baseline_folder / 'anchor.csv', 'reader': baseline_folder / 'reader.csv',
             'crop224': folder / 'crop224.csv', 'baseline': folder / 'baseline.csv',
             'multiscale_anchor': folder / 'multiscale_anchor.csv'}
    frames = {}
    for name, path in paths.items():
        record = validate_submission(path, test, sample)
        assert record['sha256'] == receipt['components'][name], name
        frames[name] = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    for name in ['anchor', 'reader']:
        assert baseline['components'][name + '.csv'] == receipt['components'][name]
    assert baseline['output'] == validate_submission(paths['baseline'], test, sample)
    assert receipt['crop_output'] == validate_submission(paths['crop224'], test, sample)
    submission = validate_submission(output / 'submission.csv', test, sample)
    assert submission == receipt['submission']
    final = pd.read_csv(output / 'submission.csv', dtype={'StudyInstanceUID': str})
    labels = final.columns[1:].tolist()
    n = len(final)
    ranks = lambda x: rankdata(x, method='average')/n
    max_error = 0.
    for label in labels:
        anchor = ranks(frames['anchor'][label].to_numpy())
        crop = ranks(frames['crop224'][label].to_numpy())
        reader = ranks(frames['reader'][label].to_numpy())
        mix = contract['crop_weight']
        spatial = ranks((1-mix)*anchor+mix*crop)
        w = contract['reader_weights'][label]
        expected = ranks((1-w)*ranks(spatial)+w*reader)
        np.testing.assert_allclose(frames['multiscale_anchor'][label], spatial, atol=1e-12, rtol=0)
        np.testing.assert_allclose(final[label], expected, atol=1e-12, rtol=0)
        bw = base_contract['weights'][label]
        np.testing.assert_allclose(frames['baseline'][label], ranks((1-bw)*anchor+bw*reader), atol=1e-12, rtol=0)
        max_error = max(max_error, float(np.max(np.abs(final[label].to_numpy()-expected))))
    pipeline = json.loads((baseline_folder / 'pipeline_preflight.json').read_text())
    assert pipeline['status'] == 'PASS_PIPELINE_AND_RECOVERY'
    assert pipeline['studies'] == base_contract['qualification_studies'] and pipeline['rows'] == 20
    assert len(pipeline['batches']) == 5 and all(b['rows'] == 4 for b in pipeline['batches'])
    assert pipeline['recovery_pixel_array_exact'] and not pipeline['labels_used']
    assert pipeline['gpu_peak_bytes'] < 14*1024**3
    log = (baseline_folder / 'reader.log').read_text()
    peak = int(re.search(r'READER_BATCH4_ALL6VIEWS_PASS peak_allocated_bytes=(\d+)', log)[1])
    assert peak < 14*1024**3
    assert f'READER_COMPLETE rows={n}' in log
    assert receipt['elapsed_seconds'] < receipt['runtime_cap'] == contract['visible_seconds_cap']
    return {'status': 'PASS_VISIBLE_TRI_READER_EXECUTION_ONLY',
            'submission': submission, 'fusion_max_abs_error': max_error,
            'elapsed_seconds': receipt['elapsed_seconds'],
            'reader_batch4_all6views_peak_bytes': peak,
            'pipeline_preflight_sha256': sha(baseline_folder / 'pipeline_preflight.json'),
            'receipt_sha256': sha(receipt_path), 'baseline_receipt_sha256': sha(baseline_folder / 'receipt.json'),
            'build_sha256': sha(RUN / 'build.json'), 'code_sha256': build['code_sha256'],
            'auditor_sha256': sha(Path(__file__)), 'accuracy_measured': False,
            'hidden_test_execution_verified': False, 'private_author0954_method_reproduced': False,
            'competition_submission_enabled': False, 'limitations': contract['limitations']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    a = p.parse_args()
    report = audit(a.output)
    (RUN / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
