"""Launch one private input ablation after evaluated prerequisites; never submit."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

from audit_e04_resource_pilot import duration_seconds
from launch_e05_kaggle_pair import ROOT, TERMINAL, inspect_missing_session, sha

RUN = ROOT / 'runs/window_integrity_20261010'
PACKAGE = ROOT / 'public_kernels/coherent_anchor_pair'
PAIR = ROOT / 'runs/e05_kaggle_pair4'
TRI = ROOT / 'runs/published_tri_reader'


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def local_gate():
    assert not (RUN / 'pair_launch.json').exists(), 'Observe the existing attempt; never push twice'
    build = read(RUN / 'pair_build.json')
    local = read(RUN / 'pair_local_check.json')
    evaluation = read(RUN / 'pair_evaluator_check.json')
    meta = read(PACKAGE / 'kernel-metadata.json')
    code = PACKAGE / meta['code_file']
    assert sha(code) == build['code_sha256'] == local['code_sha256']
    assert sha(PACKAGE / 'kernel-metadata.json') == build['metadata_sha256']
    assert local['status'] == 'PASS_PAIRED_INPUT_PACKAGE_LOCAL_GATES'
    assert evaluation['status'] == 'PASS_FROZEN_RECIPE_REPLAY_AND_PAIRED_BOOTSTRAP'
    assert sha(ROOT / 'scripts/evaluate_coherent_anchor_pair.py') == evaluation['evaluator_sha256']
    assert sha(RUN / 'pair_analysis_protocol.json') == local['analysis_protocol_sha256']
    modules = ast.literal_eval(ast.parse(code.read_text()).body[0].value)
    contract = json.loads(modules['pair_contract.json'])
    assert sha(RUN / 'ablation_protocol.json') == contract['protocol_sha256']
    assert not contract['competition_submission_enabled'] and not contract['weight_search_allowed']
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['id'] == 'adityakumarrai0401/rsna-knee-coherent-anchor-pair'
    assert meta['competition_sources'] == ['rsna-knee-abnormality-detection']
    paths = [Path(__file__), ROOT / 'scripts/evaluate_coherent_anchor_pair.py',
             ROOT / 'scripts/check_coherent_anchor_pair.py', code,
             PACKAGE / 'kernel-metadata.json', RUN / 'pair_build.json',
             RUN / 'pair_local_check.json', RUN / 'pair_evaluator_check.json',
             RUN / 'pair_analysis_protocol.json', RUN / 'ablation_protocol.json']
    prepared = {'schema': 'coherent_anchor_launch_sources_v1',
                'sources': {str(path.relative_to(ROOT)): sha(path) for path in paths},
                'platform_timeout_seconds': 2100, 'required_free_gpu_hours': 1.65,
                'maximum_new_private_launches': 1, 'competition_submission_enabled': False}
    path = RUN / 'pair_launch_preparation.json'
    if path.exists():
        assert read(path) == prepared, 'Prepared sources changed; inspect before launch'
    else:
        path.write_text(json.dumps(prepared, indent=2) + '\n')
    pending = []
    watcher = read(PAIR / 'watcher.json')
    if watcher.get('status') != 'FINISHED' or watcher.get('result') != 'BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED':
        pending.append('Finish and audit the existing E05 fold4 pair')
    result_path = TRI / 'official_result.json'
    review_path = TRI / 'official_result_evaluation.json'
    if not result_path.exists():
        pending.append('Observe the accepted N04 official result')
    else:
        result = read(result_path)
        assert result['submission_id'] == read(TRI / 'submission.json')['submission_id']
        if result['status'] != 'COMPLETE':
            pending.append('Diagnose the existing N04 evaluation failure before another launch')
        elif not review_path.exists():
            pending.append('Evaluate the N04 score against the previous best')
        else:
            review = read(review_path)
            assert review['official_result_sha256'] == sha(result_path)
            assert review['submission_id'] == result['submission_id']
            assert review['status'] == 'OFFICIAL_SCORE_EVALUATED'
    return build, meta, contract, prepared, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    build, meta, contract, prepared, pending = local_gate()
    record = {'status': 'WAIT_FOR_EVALUATED_PREREQUISITES' if pending else 'LOCAL_GATES_PASS',
              'pending': pending, 'observed_utc': now(), 'gpu_launched': False,
              'competition_submission_enabled': False}
    (RUN / 'pair_launch_preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    if not args.launch:
        print(json.dumps(record)); return
    assert not pending, 'Evaluate the existing results first'
    for arm in ['e04', 'e05_uniform']:
        subprocess.run([sys.executable, str(ROOT / 'scripts/audit_e04_fold_training.py'),
                        str(PAIR / f'output/{arm}_fold4'), '--experiment', arm,
                        '--output', str(RUN / f'prerequisite_{arm}_audit.json')],
                       check=True, stdout=subprocess.DEVNULL, timeout=120)
        audit = read(RUN / f'prerequisite_{arm}_audit.json')
        assert audit['status'] == 'PASS_PARTIAL_FOLDS_SMALL_OUTPUTS'
        assert len(audit['folds']) == 1 and audit['folds'][0]['fold'] == 4
        assert audit['folds'][0]['manifest_sha256'] == read(PAIR / f'{arm}_audit.json')['folds'][0]['manifest_sha256']
    api = KaggleApi(); api.authenticate()
    assert api.kernels_status(read(PAIR / 'launch.json')['kernel']).to_dict()['status'] == 'COMPLETE'
    history = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100) or []
    assert all(item.status.name in {'COMPLETE', 'ERROR'} for item in history), 'An existing submission is active'
    accepted = read(TRI / 'submission.json')
    matched = [item for item in history if item.ref == accepted['submission_id']]
    assert len(matched) == 1 and matched[0].status.name == 'COMPLETE'
    assert float(matched[0].public_score) == read(TRI / 'official_result.json')['public_score']
    owned, page = set(), 1
    while True:
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(row.ref for row in rows if row is not None and row.ref)
        if len(rows) < 100: break
        page += 1
        assert page <= 100
    assert meta['id'] not in owned, 'Notebook already exists; inspect its current run'
    states = []
    for ref in sorted(owned):
        time.sleep(1.25)
        try:
            status = api.kernels_status(ref).to_dict()['status']
            assert status in TERMINAL, 'An existing notebook is active: ' + ref
            states.append({'ref': ref, 'status': status})
        except HTTPError as error:
            if error.response is None or error.response.status_code != 404: raise
            states.append({'ref': ref, **inspect_missing_session(api, ref)})
        if len(states) % 20 == 0:
            print(json.dumps({'checked_existing_jobs': len(states), 'total': len(owned)}), flush=True)
    (RUN / 'pair_owned_status_scan.json').write_text(json.dumps(states, indent=2) + '\n')
    required = {meta['dataset_sources'][0]: [contract['checkpoint_filename']],
                meta['dataset_sources'][1]: ['decoder_identity.json', *contract['decoder_wheels']]}
    for ref, filenames in required.items():
        names, token, seen = set(), None, set()
        for _ in range(100):
            listing = api.dataset_list_files(ref, page_token=token, page_size=100)
            assert not listing.error_message
            names.update(Path(item.name).name for item in listing.dataset_files if item is not None)
            if set(filenames) <= names or not listing.next_page_token: break
            token = listing.next_page_token
            assert token not in seen
            seen.add(token)
        assert set(filenames) <= names, 'Required attached input is unavailable'
    quota = api.quota_view().to_dict()['gpuQuota']
    reserved = duration_seconds(quota.get('timeReserved', '0s'))
    assert reserved == 0, 'Another GPU allocation is reserved'
    available = (duration_seconds(quota['totalTimeAllowed']) - duration_seconds(quota['timeUsed']) - reserved) / 3600
    assert available >= prepared['required_free_gpu_hours'], 'Insufficient allocated GPU quota'
    # Recheck pins immediately before recording the only permitted mutation.
    for name, expected in prepared['sources'].items():
        assert sha(ROOT / name) == expected, 'Prepared sources changed during remote checks'
    receipt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': meta['id'],
               'observed_utc': now(), 'code_sha256': build['code_sha256'],
               'preparation_sha256': sha(RUN / 'pair_launch_preparation.json'),
               'available_gpu_hours': available, 'platform_timeout_seconds': 2100,
               'historical_session_limit': 'Historical404s remain unresolved; zero reservation does not establish inactivity',
               'competition_submission_enabled': False}
    path = RUN / 'pair_launch.json'
    with path.open('x') as handle:
        json.dump(receipt, handle, indent=2); handle.write('\n')
    response = api.kernels_push(str(PACKAGE), timeout='2100')
    error = bool(getattr(response, 'error', None))
    server_ref = getattr(response, 'ref', None) or meta['id']
    ref = server_ref.removeprefix('https://www.kaggle.com/').removeprefix('/').removeprefix('code/')
    receipt.update(status='PUSH_REJECTED' if error else 'PUSH_ACCEPTED',
                   kernel=ref, server_ref=server_ref, version=getattr(response, 'version_number', None),
                   error_present=error)
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    assert len(ref.split('/')) == 2 and not error, 'Inspect this accepted/rejected attempt before doing anything further'


if __name__ == '__main__':
    main()
