"""Launch the final staged E05 fold3 only after the fold4 pair is audited."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

from audit_e04_resource_pilot import duration_seconds
from launch_e05_kaggle_pair import ROOT, TERMINAL, inspect_missing_session, read, sha

RUN = ROOT / 'runs/e05_kaggle_fold3'
PAIR = ROOT / 'runs/e05_kaggle_pair4'


def now():
    return datetime.now(timezone.utc).isoformat()


def local_check():
    assert not (RUN / 'launch.json').exists(), 'Observe the existing launch; never push again'
    build = read(RUN / 'build.json')
    assert build['fold'] == 3 and not build['paired_original_control']
    assert build['status'] == 'STAGED_NOT_LAUNCHED'
    assert not build['competition_submission_enabled']
    assert build['platform_timeout_seconds'] == 27900
    for name, expected in build['sources'].items():
        assert sha(ROOT / name) == expected, 'Staged source changed: ' + name
    assert sha(ROOT / 'runs/e05_confidence_ablation/protocol.json') == build['protocol_sha256']
    package = ROOT / build['package']
    metadata = read(package / 'kernel-metadata.json')
    assert metadata['id'] == build['kernel'] and metadata['is_private'] and metadata['enable_gpu']
    assert not metadata['enable_internet'] and metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['competition_sources'] == ['rsna-knee-abnormality-detection']
    constants = {}
    for node in ast.parse((package / 'train.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'JOB_FOLDS', 'JOB_HOURS', 'WORKER_SHA256'}:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    assert constants == {'JOB_FOLDS': [3], 'JOB_HOURS': 7.25, 'WORKER_SHA256': build['worker_sha256']}
    assert not (ROOT / 'runs/e05_full_fold3/launch.json').exists(), 'Colab fold3 already launched'
    watcher = read(PAIR / 'watcher.json')
    pending = []
    if watcher.get('result') != 'BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED' or watcher['status'] != 'FINISHED':
        pending.append('Existing fold4 pair must finish and pass both independent output audits')
    else:
        for arm in ['e04', 'e05_uniform']:
            audit = read(PAIR / f'{arm}_audit.json')
            assert audit['status'] == 'PASS_PARTIAL_FOLDS_SMALL_OUTPUTS'
            assert len(audit['folds']) == 1 and audit['folds'][0]['fold'] == 4
            manifest = PAIR / f'output/{arm}_fold4/{arm}_fold4_manifest.json'
            assert sha(manifest) == audit['folds'][0]['manifest_sha256']
        # Recheck the downloaded artifacts against manifests before spending GPU time.
        import subprocess
        import sys
        for arm in ['e04', 'e05_uniform']:
            subprocess.run([sys.executable, str(ROOT / 'scripts/audit_e04_fold_training.py'),
                            str(PAIR / f'output/{arm}_fold4'), '--experiment', arm,
                            '--output', str(RUN / f'prerequisite_{arm}_audit.json')],
                           check=True, stdout=subprocess.DEVNULL, timeout=120)
    # This compute review was added after partial outcomes were known. It can
    # justify stopping further training, never promoting incomplete OOF.
    review_path = ROOT / 'runs/e05_confidence_ablation/partial_compute_review.json'
    decision_path = ROOT / 'runs/e05_confidence_ablation/last_fold_compute_decision.json'
    if not review_path.exists() or not decision_path.exists():
        pending.append('Completed fold4 evidence and an explicit further-compute decision are required')
    else:
        review, decision = read(review_path), read(decision_path)
        assert review['status'] == 'PARTIAL_COMPUTE_REVIEW_ONLY'
        assert not review['competition_submission_enabled'] and not review['complete_oof']
        assert decision['review_sha256'] == sha(review_path), 'Compute decision predates the current evidence'
        for name, expected in review['audits_sha256'].items():
            assert sha(ROOT / name) == expected, 'Reviewed audit changed: ' + name
        assert review['source_sha256'] == sha(ROOT / 'scripts/review_e05_partial_evidence.py')
        if (not review['fold4_included'] or review['folds'] != [0, 1, 2, 4]
                or decision['decision'] != 'PROCEED_WITH_FINAL_FOLD'
                or not decision.get('evidence_rationale')):
            pending.append('Further training is held until the completed partial evidence supports its cost')
    return build, package, metadata, pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    build, package, metadata, pending = local_check()
    record = {'status': 'READY' if not pending else 'WAIT_FOR_EXISTING_PAIR', 'pending': pending,
              'observed_utc': now(), 'source_dispatch_check': 'PASS',
              'launcher_sha256': sha(Path(__file__)), 'gpu_launched': False,
              'competition_submission_enabled': False}
    (RUN / 'local_preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    if not args.launch:
        print(json.dumps(record)); return
    assert not pending, 'Finish and audit the existing pair first'
    api = KaggleApi(); api.authenticate()
    pair_launch = read(PAIR / 'launch.json')
    assert api.kernels_status(pair_launch['kernel']).to_dict()['status'] == 'COMPLETE'
    for ref in metadata['kernel_sources']:
        assert api.kernels_status(ref).to_dict()['status'] == 'COMPLETE'
    owned, page = set(), 1
    while True:
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(row.ref for row in rows if row is not None and row.ref)
        if len(rows) < 100: break
        page += 1
        assert page <= 100
    assert metadata['id'] not in owned, 'Fold3 notebook already exists; inspect it instead'
    states = []
    for ref in sorted(owned):
        time.sleep(1.25)
        try:
            status = api.kernels_status(ref).to_dict()['status']
            assert status in TERMINAL, 'An existing notebook is active: ' + ref
            states.append({'ref': ref, 'status': status})
        except HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            states.append({'ref': ref, **inspect_missing_session(api, ref)})
    (RUN / 'owned_status_scan.json').write_text(json.dumps(states, indent=2) + '\n')
    quota = api.quota_view().to_dict(); gpu = quota['gpuQuota']
    # Observed live jobs can have zero reservation, so this cannot replace
    # the explicit session checks above. Historical404s remain unresolved.
    assert duration_seconds(gpu.get('timeReserved', '0s')) == 0, 'Another GPU allocation is reserved'
    available = (duration_seconds(gpu['totalTimeAllowed']) - duration_seconds(gpu['timeUsed'])) / 3600
    required = build['platform_timeout_seconds'] / 3600 + 1
    assert available >= required, 'Insufficient allocated GPU quota'
    submissions = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100)
    assert all(s.status.name in {'COMPLETE', 'ERROR'} for s in submissions), 'A submission is active'
    receipt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': metadata['id'],
               'observed_utc': now(), 'build_sha256': sha(RUN / 'build.json'),
               'code_sha256': sha(package / 'train.py'), 'launcher_sha256': sha(Path(__file__)),
               'quota': quota, 'available_gpu_hours': available,
               'platform_timeout_seconds': build['platform_timeout_seconds'],
               'historical_session_limit': '404 sessions remain unresolved; zero quota reservation does not establish inactivity',
               'competition_submission_enabled': False}
    with (RUN / 'launch.json').open('x') as handle:
        json.dump(receipt, handle, indent=2); handle.write('\n')
    response = api.kernels_push(str(package), timeout=str(build['platform_timeout_seconds']))
    error = bool(getattr(response, 'error', None))
    receipt.update(status='PUSH_REJECTED' if error else 'PUSH_ACCEPTED',
                   requested_kernel=metadata['id'], kernel=getattr(response, 'ref', None) or metadata['id'],
                   version=getattr(response, 'version_number', None),
                   kernel_id=getattr(response, 'kernel_id', None), error_present=error)
    (RUN / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    assert not error, 'Kaggle rejected this launch; inspect it before another attempt'


if __name__ == '__main__':
    main()
