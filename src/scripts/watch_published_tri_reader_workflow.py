"""One serialized path from current training evaluation to one scored candidate."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_published_tri_reader import ROOT, RUN, sha

PAIR = ROOT / 'runs/e05_kaggle_pair4'
COMPETITION = 'rsna-knee-abnormality-detection'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def timeout(*_):
    raise TimeoutError('Read observation timeout')


def main():
    lock = (RUN / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = RUN / 'workflow.json'
    if state_path.exists():
        assert read(state_path)['status'] != 'OFFICIAL_RESULT_AVAILABLE', 'Inspect the completed result'
    pinned = read(RUN / 'submission_path_preparation.json')['sources']
    api = KaggleApi()
    api.authenticate()
    signal.signal(signal.SIGALRM, timeout)
    record = {'status': 'WAIT_FOR_EXISTING_TRAINING', 'pid': os.getpid(), 'started_utc': now(),
              'current_training_kernel': read(PAIR / 'launch.json')['kernel'],
              'maximum_new_private_gpu_launches': 1, 'maximum_new_competition_submissions': 1,
              'retry_unknown_mutations': False}
    last_printed = None

    def save():
        nonlocal last_printed
        record['updated_utc'] = now()
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(state_path)
        if record['status'] != last_printed:
            print(json.dumps(record), flush=True)
            last_printed = record['status']

    def step(name, command):
        for path, expected in pinned.items():
            assert sha(ROOT / path) == expected, 'Prepared workflow source changed: ' + path
        save()
        result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=1800)
        text = result.stdout.decode('utf-8', errors='replace')
        text = re.sub(r'https?://[^\s\"\'<>]+', '[URL]', text)
        with (RUN / 'workflow_steps.log').open('a') as log:
            log.write('\n' + now() + ' ' + name + '\n' + text)
        assert result.returncode == 0, 'Step needs inspection: ' + name

    save()
    started = time.monotonic()
    while time.monotonic()-started < 10*3600:
        try:
            if not (RUN / 'launch.json').exists():
                signal.alarm(50)
                status = api.kernels_status(record['current_training_kernel']).to_dict()['status']
                signal.alarm(0)
                record.update(training_remote_status=status, training_observed_utc=now())
                assert status not in {'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_ACKNOWLEDGED'}, 'Current training failed;inspect it'
                watcher = read(PAIR / 'watcher.json')
                if status == 'COMPLETE' and watcher.get('result') == 'BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED':
                    audits = {name: read(PAIR / f'{name}_audit.json') for name in ['e04', 'e05_uniform']}
                    for report in audits.values():
                        assert report['status'] == 'PASS_PARTIAL_FOLDS_SMALL_OUTPUTS'
                        assert len(report['folds']) == 1 and report['folds'][0]['fold'] == 4
                    control, candidate = (audits[name]['folds'][0] for name in ['e04', 'e05_uniform'])
                    evaluation = {'status': 'EXISTING_PAIRED_FOLD4_EVALUATED',
                                  'control_macro_auc': control['expert_macro_auc'],
                                  'uniform_macro_auc': candidate['expert_macro_auc'],
                                  'delta': candidate['expert_macro_auc']-control['expert_macro_auc'],
                                  'expert_studies': control['expert_rows'],
                                  'audit_sha256': {name: sha(PAIR / f'{name}_audit.json') for name in audits},
                                  'owned_model_promoted': False,
                                  'scope': 'One held-out fold;fullE05fivefold comparison remains incomplete'}
                    (RUN / 'prerequisite_training_evaluation.json').write_text(json.dumps(evaluation, indent=2) + '\n')
                    record['status'] = 'CURRENT_TRAINING_EVALUATED_LAUNCHING_QUALIFICATION'
                    step('private tri-reader launch', ['scripts/launch_published_tri_reader.py', '--launch'])
                elif watcher.get('status') == 'FINISHED' and watcher.get('result') != 'BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED':
                    raise AssertionError('Existing training output needs review')
                else:
                    record['status'] = 'WAIT_FOR_EXISTING_TRAINING'
                    save()
                    time.sleep(300)
                    continue
            launch = read(RUN / 'launch.json')
            assert launch['status'] == 'PUSH_ACCEPTED', 'Launch response uncertain or rejected;reconcile the existing attempt'
            record['qualification_kernel'] = launch['kernel']
            if not (RUN / 'submission.json').exists():
                record['status'] = 'WAIT_FOR_TRI_READER_EXECUTION'
                save()
                step('observe and audit existing qualification', ['scripts/observe_published_tri_reader.py'])
                observation = read(RUN / 'observation.json')
                assert observation.get('remote_status') not in {'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_ACKNOWLEDGED'}, 'Qualification failed;inspect existing run'
                if observation.get('remote_status') == 'COMPLETE':
                    assert observation.get('audit_status') == 'PASS_VISIBLE_TRI_READER_EXECUTION_ONLY', 'Completed output did not pass independent audit'
                    record['status'] = 'QUALIFIED_SUBMITTING_ONE_VERSION'
                    step('single official version-bound submission', ['scripts/submit_published_tri_reader.py', '--submit'])
                else:
                    save()
                    time.sleep(300)
                    continue
            submission = read(RUN / 'submission.json')
            assert submission.get('submission_id') and submission['status'] == 'ACCEPTED_PENDING_EVALUATION', 'Submission response uncertain;reconcile account history'
            record.update(status='WAIT_FOR_OFFICIAL_SCORE', submission_id=submission['submission_id'])
            signal.alarm(50)
            history = api.competition_submissions(COMPETITION, page_size=100) or []
            signal.alarm(0)
            matched = [s for s in history if s.ref == submission['submission_id']]
            assert len(matched) == 1, 'Accepted submission handle absent;inspect history'
            result = matched[0]
            record.update(submission_remote_status=result.status.name, submission_observed_utc=now())
            if result.status.name in {'COMPLETE', 'ERROR'}:
                previous = [float(s.public_score) for s in history if s.ref != result.ref and s.status.name == 'COMPLETE' and s.public_score]
                score = float(result.public_score) if result.public_score else None
                report = {'status': result.status.name, 'submission_id': result.ref,
                          'kernel': submission['kernel'], 'kernel_version': submission['kernel_version'],
                          'observed_utc': now(), 'public_score': score,
                          'previous_best_public_score': max(previous) if previous else None,
                          'improvement_over_previous_best': score-max(previous) if score is not None and previous else None,
                          'private_score_known': False, 'rank_verified': False,
                          'medal_goal_complete': False}
                (RUN / 'official_result.json').write_text(json.dumps(report, indent=2) + '\n')
                record['status'] = 'OFFICIAL_RESULT_AVAILABLE'
                record['public_score'] = score
                save()
                return
            save()
        except (AssertionError, subprocess.TimeoutExpired) as e:
            signal.alarm(0)
            record.update(status='REVIEW_REQUIRED', error_type=type(e).__name__)
            save()
            return
        except Exception as e:
            signal.alarm(0)
            record['observation_error_type'] = type(e).__name__
            save()
        time.sleep(300)
    record['status'] = 'OBSERVATION_WINDOW_ENDED_RECHECK_SAME_HANDLES'
    save()


if __name__ == '__main__':
    main()
