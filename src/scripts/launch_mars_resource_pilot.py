"""Launch one bounded private pilot after reviewing the existing experiment."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

from audit_e04_resource_pilot import duration_seconds
from launch_e05_kaggle_pair import TERMINAL, inspect_missing_session, sha
from observe_coherent_anchor_pair import remote_status_name

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
PAIR = ROOT / 'runs/window_integrity_20261010'
PACKAGE = ROOT / 'public_kernels/mars_resource_pilot_v2'


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def expired(*_):
    raise TimeoutError('Observe any accepted attempt before continuing')


def local_gate():
    assert not (RUN / 'resource_launch.json').exists(), 'Observe the existing attempt, never push twice'
    build, meta = read(RUN / 'resource_build_v2.json'), read(PACKAGE / 'kernel-metadata.json')
    assert sha(PACKAGE / meta['code_file']) == build['source_sha256']
    assert sha(PACKAGE / 'kernel-metadata.json') == build['metadata_sha256']
    assert meta['id'] == 'adityakumarrai0401/rsna-knee-mars-resource-pilot-v2'
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert meta['machine_shape'] == 'NvidiaTeslaT4'
    ready = read(RUN / 'resource_ready_review.json')
    for name, expected in ready['evidence_sha256'].items():
        assert sha(RUN / name) == expected, name
    assert ready['private_images_or_patient_features_uploaded'] is False
    assert read(RUN / 'model_asset_upload.json')['status'] == 'UPLOAD_ACCEPTED'
    modules = ast.literal_eval(ast.parse((PACKAGE / meta['code_file']).read_text()).body[0].value)
    contract = json.loads(modules['pilot_contract.json'])
    protocol = json.loads(modules['resource_protocol.json'])
    assert protocol == read(RUN / 'resource_protocol_v2.json')
    assert protocol['schema'] == 'mars_rsna_resource_pilot_v2_model_only'
    assert len(protocol['studies']) == len(set(protocol['studies'])) == 24
    assert not protocol['private_image_or_embedding_upload_enabled']
    assert not contract['classification_fitting_enabled'] and not contract['competition_submission_enabled']
    assert contract['platform_timeout_seconds'] == 1260
    for name, expected in contract['module_sha256'].items():
        assert sha_text(modules[name]) == expected, name
        if (ROOT / 'inference' / name).exists():
            assert sha(ROOT / 'inference' / name) == expected, name
    controls = [Path(__file__), ROOT / 'scripts/audit_mars_resource_pilot.py',
                ROOT / 'scripts/observe_mars_resource_pilot.py',
                ROOT / 'scripts/observe_coherent_anchor_pair.py',
                ROOT / 'scripts/audit_e04_resource_pilot.py',
                ROOT / 'scripts/launch_e05_kaggle_pair.py',
                PACKAGE / meta['code_file'], PACKAGE / 'kernel-metadata.json',
                RUN / 'resource_build_v2.json', RUN / 'resource_protocol_v2.json']
    prepared = {'schema': 'mars_resource_control_sources_v2',
                'sources_sha256': {str(p.relative_to(ROOT)): sha(p) for p in controls},
                'competition_submission_enabled': False, 'maximum_new_private_runs': 1}
    prepared_path = RUN / 'resource_control_preparation.json'
    if prepared_path.exists():
        assert read(prepared_path) == prepared, 'Review changed controls before launch'
    else:
        prepared_path.write_text(json.dumps(prepared, indent=2) + '\n')
    pending = []
    evaluation_path = PAIR / 'pair_evaluation.json'
    review_path = PAIR / 'pair_followup_review.json'
    if not evaluation_path.exists():
        pending.append('Finish and evaluate the existing coherent-window comparison')
    else:
        evaluation = read(evaluation_path)
        assert evaluation['status'] == 'PAIRED_INPUT_ACCURACY_DIAGNOSTIC_COMPLETE'
        execution = read(PAIR / 'pair_execution_audit.json')
        assert execution['status'] == 'PASS_PAIRED_EXECUTION_AND_INPUT_IDENTITY'
        assert evaluation['execution_audit'] == execution
        if not review_path.exists():
            pending.append('Review primary N02 accuracy, uncertainty, target changes and control drift')
        else:
            review = read(review_path)
            assert review['evaluation_sha256'] == sha(evaluation_path)
            assert review['primary_protocol_sha256'] == sha(PAIR / 'pair_decision_protocol.json')
            assert review['decision'] == 'PROCEED_WITH_MARS_RESOURCE_PILOT'
            assert isinstance(review['rationale'], str) and len(review['rationale'].strip()) >= 40
    return build, meta, contract, pending


def sha_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, expired)
    attempt = None
    try:
        build, meta, contract, pending = local_gate()
        preflight = {'status': 'WAIT_FOR_EVALUATED_CURRENT_EXPERIMENT' if pending else 'LOCAL_LAUNCH_GATES_PASS',
                     'observed_utc': now(), 'pending': pending, 'gpu_launched': False,
                     'competition_submission_enabled': False}
        (RUN / 'resource_launch_preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
        if pending or not args.launch:
            print(json.dumps(preflight)); return
        api = KaggleApi(); api.authenticate()
        signal.alarm(450)
        assert remote_status_name(api.kernels_status(read(PAIR / 'pair_launch.json')['kernel'])) == 'COMPLETE'
        history = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100) or []
        assert all(row.status.name in {'COMPLETE', 'ERROR'} for row in history), 'An official submission is active'
        owned = set()
        for page in range(1, 101):
            rows = api.kernels_list(mine=True, page=page, page_size=100) or []
            owned.update(row.ref for row in rows if row and row.ref)
            if len(rows) < 100: break
        else:
            raise AssertionError('Owned notebook pagination did not finish')
        assert meta['id'] not in owned, 'Observe the existing notebook instead'
        states = []
        for ref in sorted(owned):
            time.sleep(1.25)
            try:
                status = remote_status_name(api.kernels_status(ref))
                assert status in TERMINAL, 'An owned notebook is active or unknown'
                states.append({'ref': ref, 'status': status})
            except HTTPError as error:
                if error.response is None or error.response.status_code != 404: raise
                states.append({'ref': ref, **inspect_missing_session(api, ref)})
            if len(states) % 20 == 0:
                print(json.dumps({'checked_existing_notebooks': len(states), 'total': len(owned)}), flush=True)
        (RUN / 'resource_owned_status_scan.json').write_text(json.dumps(states, indent=2) + '\n')
        for dataset in meta['dataset_sources']:
            assert api.dataset_status(dataset) == 'ready'
        manifest = read(RUN / 'model_asset_package.json')
        listing = api.dataset_list_files(meta['dataset_sources'][0], page_size=100)
        assert not listing.error_message and not listing.next_page_token
        remote_sizes = {row.name: row.total_bytes for row in listing.dataset_files if row}
        assert remote_sizes == {name: entry['bytes'] for name, entry in manifest['files'].items()
                                if name != 'dataset-metadata.json'}
        quota = api.quota_view().to_dict()['gpuQuota']
        reserved = duration_seconds(quota.get('timeReserved', '0s'))
        available = (duration_seconds(quota['totalTimeAllowed']) - duration_seconds(quota['timeUsed']) - reserved) / 3600
        assert reserved == 0 and available >= 1.7, 'Insufficient free GPU capacity including reserve'
        # Repeat all local identities after the remote preflight.
        _, _, _, pending = local_gate()
        assert not pending
        attempt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': meta['id'],
                   'observed_utc': now(), 'code_sha256': build['source_sha256'],
                   'metadata_sha256': build['metadata_sha256'],
                   'resource_protocol_sha256': build['resource_protocol_sha256'],
                   'control_preparation_sha256': sha(RUN / 'resource_control_preparation.json'),
                   'current_experiment_review_sha256': sha(PAIR / 'pair_followup_review.json'),
                   'available_gpu_hours': available, 'platform_timeout_seconds': 1260,
                   'historical_session_limit': 'Historical404s remain unresolved; zero reservation does not prove their inactivity',
                   'competition_submission_enabled': False, 'classification_fitting_enabled': False}
        with (RUN / 'resource_launch.json').open('x') as handle:
            handle.write(json.dumps(attempt, indent=2) + '\n')
        signal.alarm(60)
        response = api.kernels_push(str(PACKAGE), timeout='1260')
        error_present = bool(getattr(response, 'error', None))
        ref = getattr(response, 'ref', None) or meta['id']
        ref = ref.removeprefix('https://www.kaggle.com/').removeprefix('/').removeprefix('code/')
        attempt.update(status='PUSH_REJECTED' if error_present else 'PUSH_ACCEPTED',
                       kernel=ref, version=getattr(response, 'version_number', None), error_present=error_present)
        assert not error_present and isinstance(attempt['version'], int) and attempt['version'] > 0
    except Exception as error:
        if attempt is None:
            print(json.dumps({'status': 'PREFLIGHT_REVIEW_REQUIRED_NO_PUSH', 'error_type': type(error).__name__}))
        else:
            attempt['observation_error_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        if attempt is not None:
            (RUN / 'resource_launch.json').write_text(json.dumps(attempt, indent=2) + '\n')
            print(json.dumps(attempt))


if __name__ == '__main__':
    main()
