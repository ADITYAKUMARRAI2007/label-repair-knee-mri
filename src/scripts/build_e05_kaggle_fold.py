"""Stage one E05 candidate on Kaggle using the already verified training recipe."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=[2, 3, 4], default=4)
    parser.add_argument('--paired-control', action='store_true',
                        help='Use the second T4 for the missing original control fold4')
    args = parser.parse_args()
    assert not args.paired_control or args.fold == 4, 'Only control fold4 is missing'
    tag = f'pair{args.fold}' if args.paired_control else f'fold{args.fold}'
    run = ROOT / f'runs/e05_kaggle_{tag}'
    assert not (run / 'launch.json').exists(), 'Preserve an existing launch'
    run.mkdir(exist_ok=True)
    read = lambda name: json.loads((ROOT / name).read_text())
    first = read('runs/e05_full_fold0/watcher.json')
    assert first['result'] == 'COMPLETE' and first['fold_audit_passed']
    assert read('runs/e05_full_fold0/verified_final/integrity.json')['all_member_bytes_independently_verified']
    assert read('runs/e05_confidence_ablation/control_integrity.json')['status'] == 'PASS_ALL_FOUR'
    timing = read('runs/E04_contract/first_pair_epoch_budget_audit.json')
    assert timing['status'] == 'PASS_COMPLETE_EPOCH_TIMING'
    hours = 7.25
    assert timing['projection_with_15_percent_margin_hours'] <= hours
    original = ROOT / 'runs/e05_confidence_ablation/full_worker.py'
    worker = original.read_text()
    assert sha(original) == read('runs/e05_confidence_ablation/full_worker_build.json')['code_sha256']
    assert sha(original) == read('runs/e05_full_fold0/job.json')['files_sha256']['worker.py']
    # Numerical code, seed derivation, folds and final-EMA selection are identical.
    worker = worker.replace('/content/kaggle/', '/kaggle/')
    assert worker.replace('/kaggle/', '/content/kaggle/') == original.read_text()
    compile(worker, 'worker.py', 'exec')
    worker_sha = hashlib.sha256(worker.encode()).hexdigest()
    supervisor_path = ROOT / 'training/e04_fold_supervisor.py'
    assert sha(supervisor_path) == 'efad8b52161de249b3a61a29c17a7b56c28c6254670a7c9adca4ea469cc5daee'
    supervisor = supervisor_path.read_text()
    supervisor = replace_once(supervisor, '# __FROZEN_JOB__',
                              f'JOB_FOLDS = [{args.fold}]\nJOB_HOURS = {hours}')
    supervisor = replace_once(supervisor, '# __WORKER_SOURCE__',
                              f'WORKER_SOURCE = {worker!r}\nWORKER_SHA256 = {worker_sha!r}')
    supervisor = replace_once(supervisor, '    started = time.time()', '''    started = time.time()
    import timm, numpy
    if (torch.__version__ != '2.11.0+cu128' or timm.__version__ != '1.0.29'
            or numpy.__version__ != '2.1.3'):
        raise RuntimeError('Kaggle environment differs from the verified T4 controls')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Measured Kaggle budget requires a T4')
    os.environ['E05_ARM'] = 'uniform'
    # Kaggle persists /kaggle/working at completion; Colab upload leases are absent.
    os.environ.pop('E05_UPLOAD_LEASES', None)''')
    # Modify the supervisor, not the embedded worker, when naming output folders.
    embedded = f'WORKER_SOURCE = {worker!r}\nWORKER_SHA256 = {worker_sha!r}'
    supervisor = replace_once(supervisor, embedded, '# __EMBEDDED_E05_WORKER__')
    supervisor = supervisor.replace('e04_full_fold_worker.py', 'e05_full_fold_worker.py')
    supervisor = supervisor.replace('e04_fold', 'e05_uniform_fold')
    supervisor = supervisor.replace('e04_independent_fold_job_v1', 'e05_uniform_kaggle_fold_job_v1')
    control_path = ROOT / 'public_kernels/e04_convnext_folds_4/e04_full_fold_worker_preview.py'
    if args.paired_control:
        assert sha(control_path) == '082cd85baed7bf293983e86b572e7f42347a008f350756264c34f69346212a80'
        supervisor = replace_once(supervisor, f'JOB_FOLDS = [{args.fold}]',
                                  f'JOB_FOLDS = [{args.fold}, {args.fold}]')
        supervisor = replace_once(supervisor, 'def main():', '''def output_prefix(rank, fold):
    return f'e04_fold{fold}' if rank == 0 else f'e05_uniform_fold{fold}'


def main():''')
        supervisor = replace_once(supervisor, '            directory = out / f"e05_uniform_fold{fold}"', '''            directory = out / output_prefix(rank, fold)
            fold_worker_path = out / f'paired_worker_{rank}.py'
            source = CONTROL_WORKER_SOURCE if rank == 0 else WORKER_SOURCE
            expected = CONTROL_WORKER_SHA256 if rank == 0 else WORKER_SHA256
            fold_worker_path.write_text(source)
            if sha(fold_worker_path) != expected:
                raise RuntimeError('Paired worker identity changed')''')
        supervisor = replace_once(supervisor, '            env = os.environ.copy()',
                                  "            env = os.environ.copy()\n            env['E05_ARM'] = 'control' if rank == 0 else 'uniform'")
        supervisor = replace_once(supervisor, 'str(worker_path)], env=env,',
                                  'str(fold_worker_path)], env=env,')
        supervisor = replace_once(supervisor,
            'states = [last_progress(out / f"e05_uniform_fold{f}/worker.log") for f in JOB_FOLDS]',
            'states = [last_progress(out / output_prefix(rank, f) / "worker.log") for rank, f in enumerate(JOB_FOLDS)]')
        supervisor = replace_once(supervisor,
            '    for fold in JOB_FOLDS:\n        path = out / f"e05_uniform_fold{fold}/e05_uniform_fold{fold}_manifest.json"',
            '    for rank, fold in enumerate(JOB_FOLDS):\n        prefix = output_prefix(rank, fold)\n        path = out / prefix / f"{prefix}_manifest.json"')
        supervisor = replace_once(supervisor,
            'results.append({"fold": fold, "status": result["status"],',
            'results.append({"fold": fold, "arm": "control" if rank == 0 else "uniform", "status": result["status"],')
        supervisor = supervisor.replace('e05_uniform_kaggle_fold_job_v1', 'e05_paired_kaggle_fold_job_v1')
        supervisor = supervisor.replace('e05_uniform_fold_job_manifest.json', 'e05_paired_fold_job_manifest.json')
        supervisor = replace_once(supervisor, '# __EMBEDDED_E05_WORKER__',
            '# __EMBEDDED_E05_WORKER__\n' + f'CONTROL_WORKER_SOURCE = {control_path.read_text()!r}\nCONTROL_WORKER_SHA256 = {sha(control_path)!r}')
    supervisor = replace_once(supervisor, '# __EMBEDDED_E05_WORKER__', embedded)
    compile(supervisor, 'train.py', 'exec')
    package = ROOT / f'public_kernels/e05_{"paired" if args.paired_control else "uniform"}_fold{args.fold}'
    package.mkdir(exist_ok=True)
    (package / 'train.py').write_text(supervisor)
    (package / 'worker_preview.py').write_text(worker)
    metadata = read('public_kernels/e04_convnext_folds_4/kernel-metadata.json')
    metadata.update(id=f'adityakumarrai0401/rsna-knee-e05-uniform-fold-{args.fold}',
                    title=f'RSNA Knee E05 Uniform Fold {args.fold}', code_file='train.py')
    if args.paired_control:
        metadata.update(id=f'adityakumarrai0401/rsna-knee-e05-paired-fold-{args.fold}',
                        title=f'RSNA Knee E05 Control and Uniform Fold {args.fold}')
    assert metadata['is_private'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    (package / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result = {'status': 'STAGED_NOT_LAUNCHED', 'fold': args.fold, 'arm': 'uniform',
              'paired_original_control': args.paired_control,
              'package': str(package.relative_to(ROOT)), 'kernel': metadata['id'],
              'original_colab_worker_sha256': sha(original), 'worker_sha256': worker_sha,
              'training_code_identical_after_storage_path_normalization': True,
              'sources': {str(p.relative_to(ROOT)): sha(p) for p in
                          [Path(__file__), supervisor_path, package / 'train.py',
                           package / 'worker_preview.py', package / 'kernel-metadata.json']},
              'protocol_sha256': sha(ROOT / 'runs/e05_confidence_ablation/protocol.json'),
              'job_budget_hours': hours, 'platform_timeout_seconds': 27900,
              'fixed_final_ema_epoch': 14, 'competition_submission_enabled': False,
              'launch_requirements': ['Current Colab fold complete and independently audited',
                                      'Fresh quota and owned-job observations',
                                      'No existing run for this candidate fold',
                                      'Platform timeout and pre-write launch receipt'],
              'limitations': ['Remaining candidate hardware may mix A100/CUDA13.0 and T4/CUDA12.8',
                              'Kaggle checkpoint persistence is only confirmed after job completion']}
    (run / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    if args.paired_control:
        result['control_worker_sha256'] = sha(control_path)
        result['sources'][str(control_path.relative_to(ROOT))] = sha(control_path)
        (run / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
        previous = ROOT / f'runs/e05_kaggle_fold{args.fold}/build.json'
        if previous.exists():
            assert not (previous.parent / 'launch.json').exists()
            old = json.loads(previous.read_text())
            old.update(status='SUPERSEDED_BY_PAIRED_PACKAGE', replacement=str(run.relative_to(ROOT)))
            previous.write_text(json.dumps(old, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'fold', 'worker_sha256',
                      'training_code_identical_after_storage_path_normalization',
                      'platform_timeout_seconds', 'competition_submission_enabled']}))


if __name__ == '__main__':
    main()
