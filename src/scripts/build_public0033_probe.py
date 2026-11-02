#!/usr/bin/env python3
"""Freeze one private specialist parity/validation diagnostic."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit_path = ROOT / 'runs/public0033_source_audit.json'
    audit = json.loads(audit_path.read_text())
    for relative, expected in audit['sources'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('audited source changed: ' + relative)
    check_path = ROOT / 'runs/specialist_cache_capture_check.json'
    if json.loads(check_path.read_text())['status'] != 'PASS_LOCAL_MEMMAP_CAPTURE_CHECKS':
        raise RuntimeError('local capture gate missing')
    original = ROOT / 'research/new_public/public0033_original'
    text = (original / 'source_for_review.py').read_text()
    start = text.index('from __future__ import annotations')
    stop = text.index('class SlotHead(nn.Module):', start)
    preprocessing = text[start:stop]
    old = '    for study in os.scandir(base):\n        if study.is_dir():'
    new = '    for study in os.scandir(base):\n        if study.is_dir() and (STUDY_SELECTION is None or study.name in STUDY_SELECTION):'
    if preprocessing.count(old) != 1:
        raise RuntimeError('original scanner structure changed')
    preprocessing = preprocessing.replace(old, new) + '\nSTUDY_SELECTION = None\n'
    split = json.loads((ROOT / 'runs/E03_contract/split.json').read_text())
    with (original / 'reference/public0033_bag_raw.csv').open(newline='') as handle:
        reference = list(csv.DictReader(handle))
    targets = json.loads((ROOT / 'runs/grade_supervision_trial_v1/contract.json').read_text())['target_order']
    receipt = json.loads((original / 'reference/public0033_cached_inference_receipt.json').read_text())
    contract = {'schema': 'public0033_isolated_probe_v1', 'targets': targets,
                'input_config': audit['input_config'], 'expert_ids': split['gold'],
                'report_connected_expert_ids': audit['report_connected_expert_ids'],
                'reference_predictions': reference, 'reference_mask_sha256': receipt['mask_sha256'],
                'reference_tolerance': 1e-6, 'checkpoint_sha256': audit['checkpoint_sha256'],
                'runtime_sha256': receipt['runtime_sha256'],
                'official_file_sha256': {name: sha(ROOT / 'data' / name) for name in ['train.csv', 'train_series.csv']},
                'evaluation_rule': 'Reject parity/fallback/incomplete failures. Full58 macro>=0.90 is necessary but not sufficient; disclose known linked expert and separately report57. Never fit blend weights, choose epochs, or retrain against this diagnostic. Only propose next action after audit.',
                'validation_limitations': audit['evaluation_limitations'],
                'gpu_budget_hours_including_startup_margin': 1.5,
                'competition_submission_enabled': False, 'model_training_enabled': False,
                'preprocessing_edit': 'filter header scanning to fixed selected studies; pixel preparation unchanged',
                'original_preprocessing_source_sha256': hashlib.sha256(text[start:stop].encode()).hexdigest(),
                'source_audit_sha256': sha(audit_path)}
    worker_path = ROOT / 'training/public0033_probe.py'
    capture_path = ROOT / 'training/specialist_cache_capture.py'
    modules = {'specialist_preprocess.py': preprocessing, 'specialist_cache_capture.py': capture_path.read_text(),
               'public0033_probe.py': worker_path.read_text().replace('# __PROBE_CONTRACT__', 'CONTRACT = ' + repr(contract))}
    for name, code in modules.items():
        compile(code, name, 'exec')
    directory = ROOT / 'public_kernels/public0033_isolated_probe_v1'
    directory.mkdir(exist_ok=True)
    if (ROOT / 'runs/public0033_probe_launch.json').exists():
        raise RuntimeError('preserve launched source; observe original handle')
    code_path = directory / 'public0033-isolated-probe.py'
    wrapper = '''import os, signal, subprocess, sys
from pathlib import Path
MODULES = ''' + repr(modules) + '''
root=Path('/kaggle/working/public0033_probe_source');root.mkdir(exist_ok=True)
for name,code in MODULES.items():(root/name).write_text(code)
p=subprocess.Popen([sys.executable,str(root/'public0033_probe.py')],start_new_session=True)
try:code=p.wait(timeout=3600)
except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
if code:raise RuntimeError(f'bounded specialist diagnostic failed; child exit{code}')
'''
    compile(wrapper, code_path.name, 'exec')
    code_path.write_text(wrapper)
    metadata = {'id': 'adityakumarrai0401/rsna-knee-public0033-isolated-probe',
                'title': 'RSNA Knee Public0033 Isolated Diagnostic', 'code_file': code_path.name,
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': ['renta0426/rsna-knee-public0033-meniscus-bag-v1'],
                'kernel_sources': [], 'model_sources': [],
                'competition_sources': ['rsna-knee-abnormality-detection']}
    metadata_path = directory / 'kernel-metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    build = {'status': 'BUILT_NOT_LAUNCHED', 'contract': contract,
             'sources': {str(p.relative_to(ROOT)): sha(p) for p in [worker_path, capture_path, code_path, metadata_path]},
             'gates': {str(p.relative_to(ROOT)): sha(p) for p in [audit_path, check_path]}}
    (ROOT / 'runs/public0033_probe_source.json').write_text(json.dumps(build, indent=2) + '\n')
    print(json.dumps({'status': build['status'], 'kernel': metadata['id'], 'gpu_cap_hours': 1.5,
                      'parity_first': True, 'expert_count': 58, 'competition_submission': False}))


if __name__ == '__main__':
    main()
