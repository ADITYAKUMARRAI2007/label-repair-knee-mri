"""Build the full-corpus notebook only after numerical/cost review exists."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'
PACKAGE = ROOT / 'public_kernels/mars_full_corpus'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    approval = BASE / 'corpus_resource_review.json'
    if not approval.exists():
        print(json.dumps({'status': 'WAIT_FOR_MEASURED_PILOT_PARITY_AND_COST_REVIEW',
                          'new_gpu_jobs': 0, 'new_competition_submissions': 0})); return
    review = json.loads(approval.read_text())
    assert review['decision'] == 'APPROVE_MEASURED_MARS_FULL_FEATURE_EXTRACTION'
    assert review['numerical_pilot_passed'] and review['complete_worker_cost_reviewed']
    assert review['required_studies'] == 4407 and not review['automatic_classifier_training']
    assert not (BASE / 'corpus_launch.json').exists(), 'Preserve any accepted or uncertain run'
    old = json.loads((ROOT / 'public_kernels/mars_resource_cuda_fix/pilot.ipynb').read_text())
    code = next(c['source'] for c in old['cells'] if c['cell_type'] == 'code')
    prior = ast.literal_eval(ast.parse(code).body[0].value)
    old_contract = json.loads(prior['pilot_contract.json'])
    names = ['mars_encoder.py', 'mars_legacy_merging.py', 'mars_feature_cache.py',
             'mars_volume.py', 'mars_study.py', 'orthodiffusion_volume.py', 'check_decoders.py']
    modules = {n: prior[n] for n in names}
    for name in names:
        if (ROOT / 'inference' / name).exists():
            assert (ROOT / 'inference' / name).read_text() == modules[name]
    modules.update({n: (ROOT / 'inference' / n).read_text()
                    for n in ['mars_corpus_extract.py', 'mars_corpus_driver.py']})
    modules['corpus_review.json'] = approval.read_text()
    contract = {k: old_contract[k] for k in ['torch_version', 'asset_sha256', 'decoder_wheels', 'monai_wheel']}
    contract.update(schema='mars_complete_frozen_corpus_v1',
                    maximum_parent_seconds=review['maximum_parent_seconds'],
                    platform_timeout_seconds=review['platform_timeout_seconds'],
                    classifier_training_started=False, competition_submission_enabled=False,
                    module_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in modules.items()})
    assert review['maximum_worker_seconds'] < review['maximum_parent_seconds'] < review['platform_timeout_seconds']
    modules['corpus_contract.json'] = json.dumps(contract, indent=2) + '\n'
    code = 'MODULES=' + repr(modules) + '\n' + (
        "exec(compile(MODULES['mars_corpus_driver.py'], 'mars_corpus_driver.py', 'exec'))\n")
    ast.parse(code)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': old['metadata'],
                'cells': [{'cell_type': 'code', 'id': 'corpus-driver', 'metadata': {},
                           'execution_count': None, 'outputs': [], 'source': code}]}
    meta = json.loads((ROOT / 'public_kernels/mars_resource_cuda_fix/kernel-metadata.json').read_text())
    meta.pop('id_no', None)
    meta.update(id='adityakumarrai0401/rsna-knee-mars-full-frozen-corpus',
                title='RSNA Knee MARS Full Frozen Corpus')
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE / 'pilot.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    record = {'status': 'FULL_CORPUS_NOTEBOOK_PACKAGED_NOT_LAUNCHED',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'code_cell_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'notebook_sha256': sha(PACKAGE / 'pilot.ipynb'),
              'metadata_sha256': sha(PACKAGE / 'kernel-metadata.json'),
              'review_sha256': sha(approval), 'new_gpu_jobs': 0,
              'new_competition_submissions': 0}
    (BASE / 'corpus_build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
