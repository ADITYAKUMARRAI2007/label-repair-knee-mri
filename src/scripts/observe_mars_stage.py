"""Collect native stage receipts; feature arrays remain in the Kaggle corpus."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from observe_coherent_anchor_pair import remote_status_name
from launch_mars_stage import ROOT, BASE, STAGES, read, sha
from resume_mars_pilot_collection import collect

TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}
PATTERNS = {
    'corpus': (r'^(submission\.csv|mars_latest_corpus_attempt\.json|.*\.log|'
               r'mars_corpus_attempt_[^/]+/(failure\.json|submission\.csv|source/[^/]+|'
               r'corpus/(corpus_receipt\.json|shard[01]_(receipt|failure)\.json)))$'),
    'classifier': (r'^(submission\.csv|mars_latest_classifier_attempt\.json|.*\.log|'
                   r'mars_classifier_attempt_[^/]+/(failure\.json|execution_receipt\.json|submission\.csv|'
                   r'training/((training_receipt|fold_receipts|history|feature_inputs|heldout_replay)\.json|'
                   r'mars_oof_predictions\.csv|mars_head_fold[0-4]\.pt)))$'),
}


def audit(stage, output, modules, approval):
    assert not (output / 'failure.json').exists()
    assert not list(output.rglob('submission.csv'))
    if stage == 'corpus':
        for name, source in modules.items():
            assert (output / 'source' / name).read_text() == source, name
        contract = json.loads(modules['corpus_contract.json'])
        receipt_path = output / 'corpus/corpus_receipt.json'
        receipt = read(receipt_path)
        assert receipt['status'] == 'COMPLETE_4407_STUDY_FROZEN_MARS_FEATURE_CORPUS'
        assert receipt['contract_sha256'] == hashlib.sha256(modules['corpus_contract.json'].encode()).hexdigest()
        assert receipt['review_sha256'] == sha(BASE / 'corpus_resource_review.json')
        assert receipt['extraction_contract_sha256'] == approval['extraction_contract_sha256']
        assert 0 < receipt['elapsed_seconds'] <= contract['maximum_parent_seconds']
        assert receipt['studies'] == 4407 and receipt['label_values_used'] is False
        assert receipt['classifier_training_started'] is False and not receipt['competition_submission_enabled']
        records = []
        for shard in range(2):
            path = output / f'corpus/shard{shard}_receipt.json'
            assert sha(path) == receipt['shard_receipts_sha256'][shard]
            item = read(path)
            assert item['status'] == 'COMPLETE_MARS_FEATURE_SHARD' and item['shard'] == shard
            assert item['shard_count'] == 2 and len(item['records']) == item['required_studies']
            assert item['review_sha256'] == receipt['review_sha256']
            assert item['extraction_contract_sha256'] == receipt['extraction_contract_sha256']
            assert 0 < item['elapsed_seconds'] <= approval['maximum_worker_seconds']
            assert 0 < item['gpu_peak_bytes'] <= 16 * (1 << 30)
            assert item['label_values_used'] is False and not item['classifier_training_started']
            records.extend(item['records'])
        ids = sorted(row['study_id'] for row in records)
        assert len(ids) == len(set(ids)) == 4407
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == approval['ordered_study_ids_sha256']
        for shard in range(2):
            assert [row['study_id'] for row in read(output / f'corpus/shard{shard}_receipt.json')['records']] == ids[shard::2]
        for row in records:
            assert 0 < row['seconds'] and row['cache_reused'] is False
            for key in ['cache_sha256', 'feature_sha256', 'input_sha256']:
                assert len(row[key]) == 64 and all(c in '0123456789abcdef' for c in row[key])
        return {'status': 'PASS_COMPLETE_NATIVE_CORPUS_EXECUTION_AND_COVERAGE',
                'studies': 4407, 'corpus_receipt_sha256': sha(receipt_path),
                'extraction_contract_sha256': receipt['extraction_contract_sha256'],
                'elapsed_seconds': receipt['elapsed_seconds'],
                'local_feature_arrays_downloaded_or_independently_replayed': False,
                'array_integrity_checked_by_source_matched_native_driver': True,
                'independent_native_array_recheck_required_before_classifier_updates': True,
                'clinical_accuracy_measured': False, 'competition_submission_enabled': False}
    contract = json.loads(modules['classifier_contract.json'])
    receipt = read(output / 'execution_receipt.json')
    assert receipt['status'] == 'BOUNDED_COMPLETE_FIVE_FOLD_CLASSIFIER_EXECUTION'
    assert receipt['contract_sha256'] == hashlib.sha256(modules['classifier_contract.json'].encode()).hexdigest()
    assert receipt['resource_approval_sha256'] == sha(BASE / 'classifier_resource_approval.json')
    assert receipt['corpus_receipt_sha256'] == approval['corpus_receipt_sha256']
    assert receipt['training_receipt_sha256'] == sha(output / 'training/training_receipt.json')
    replay_path = output / 'training/heldout_replay.json'
    replay = read(replay_path)
    assert sha(replay_path) == receipt['heldout_replay_sha256']
    assert replay['status'] == 'PASS_ALL_4407_RELOADED_HELDOUT_HEAD_PREDICTIONS'
    assert replay['studies'] == 4407 and 0 <= replay['maximum_absolute_error'] <= 1e-5
    assert replay['absolute_tolerance'] == 1e-5
    assert replay['training_receipt_sha256'] == receipt['training_receipt_sha256']
    assert replay['oof_predictions_sha256'] == sha(output / 'training/mars_oof_predictions.csv')
    assert replay['serving_protocol_sha256'] == hashlib.sha256(modules['serving_protocol.json'].encode()).hexdigest()
    assert replay['head_source_sha256'] == hashlib.sha256(modules['inference/mars_head_inference.py'].encode()).hexdigest()
    assert replay['verifier_source_sha256'] == hashlib.sha256(modules['scripts/replay_mars_heldout_heads.py'].encode()).hexdigest()
    assert replay['extraction_contract_sha256'] == approval['extraction_contract_sha256']
    assert not replay['test_ensemble_or_end_to_end_dicom_serving_verified']
    assert 0 < receipt['elapsed_seconds'] <= contract['maximum_parent_seconds']
    assert not receipt['competition_submission_enabled'] and not receipt['automatic_blend_search']
    return {'status': 'PASS_NATIVE_FIXED_CLASSIFIER_EXECUTION_CHAIN',
            'execution_receipt_sha256': sha(output / 'execution_receipt.json'),
            'corpus_receipt_sha256': approval['corpus_receipt_sha256'],
            'heldout_replay_sha256': sha(replay_path),
            'heldout_head_predictions_replayed': 4407,
            'training_output': str((output / 'training').relative_to(ROOT)),
            'clinical_accuracy_requires_separate_evaluator': True, 'competition_submission_enabled': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    args = parser.parse_args(); stage = args.stage
    launch_path = BASE / f'{stage}_launch.json'
    if not launch_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_ACCEPTED_STAGE', 'stage': stage, 'gpu_launched': False})); return
    launch = read(launch_path)
    approval_name, build_name, package_name, decision = STAGES[stage]
    assert sha(BASE / build_name) == launch['build_sha256']
    assert sha(BASE / approval_name) == launch['review_sha256']
    approval = read(BASE / approval_name)
    assert approval['decision'] == decision
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
              'kernel': launch['kernel'], 'gpu_launched': False, 'competition_submission_enabled': False}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    try:
        api = KaggleApi(); api.authenticate(); signal.alarm(45)
        record['remote_status'] = remote_status_name(api.kernels_status(launch['kernel']))
        if record['remote_status'] in TERMINAL:
            destination = BASE / f'{stage}_remote_source'; signal.alarm(60)
            api.kernels_pull(launch['kernel'], str(destination), metadata=True, quiet=True)
            meta = read(destination / 'kernel-metadata.json')
            notebook = read(destination / meta['code_file'])
            cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
            assert len(cells) == 1
            code = cells[0]['source']; code = ''.join(code) if isinstance(code, list) else code
            assert hashlib.sha256(code.encode()).hexdigest() == launch['code_cell_sha256']
            modules = ast.literal_eval(ast.parse(code).body[0].value)
            record['source_matches_accepted'] = True
            output_root = BASE / f'{stage}_output'
            completed, failures = collect(api, launch['kernel'], output_root, PATTERNS[stage],
                                          BASE / f'{stage}_execution.log')
            record['collected_files'] = completed
            record['download_failures'] = failures
            if failures:
                raise RuntimeError('Some artifacts require collection from the same terminal run')
            assert not list(output_root.rglob('submission.csv'))
            marker = read(output_root / f'mars_latest_{stage}_attempt.json')
            name = marker['attempt_directory']
            assert Path(name).name == name and name.startswith(f'mars_{stage}_attempt_')
            output = output_root / name
            record['attempt_directory'] = name
            if record['remote_status'] == 'COMPLETE':
                reviewed = audit(stage, output, modules, approval)
                (BASE / f'{stage}_execution_audit.json').write_text(json.dumps(reviewed, indent=2) + '\n')
                record['execution_audit_status'] = reviewed['status']
                if stage == 'classifier':
                    signal.alarm(120)
                    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_mars_classifier.py'),
                                    '--training-output', str(output / 'training'),
                                    '--output', str(BASE / 'classifier_evaluation.json')], check=True, timeout=90)
                    record['evaluation_decision'] = read(BASE / 'classifier_evaluation.json')['decision']
            record['next_action'] = 'REVIEW_THIS_STAGE_BEFORE_ANY_NEXT_GPU_JOB_OR_SUBMISSION'
    except Exception as error:
        record['observation_error_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        (BASE / f'{stage}_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
