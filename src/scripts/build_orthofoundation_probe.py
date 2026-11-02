"""Freeze one encoder hypothesis and build the offline resource-gated notebook."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (ROOT / 'runs/orthofoundation_probe_launch.json').exists():
        raise RuntimeError('preserve launched source')
    trial_path = ROOT / 'runs/orthofoundation_trial_v1/contract.json'
    if trial_path.exists():
        raise RuntimeError('trial already frozen; inspect before any rebuild')
    package_path = ROOT / 'runs/orthofoundation_asset_package.json'
    package = json.loads(package_path.read_text())
    if package['status'] != 'READY_PRIVATE_UPLOAD':
        raise RuntimeError('assets not verified')
    provenance = json.loads((ROOT / 'datasets/orthofoundation_assets_v1/provenance.json').read_text())
    if sha(ROOT / 'training/orthofoundation_encoder.py') != provenance['adapter_sha256']:
        raise RuntimeError('verified adapter changed')
    trial = {
        'schema': 'orthofoundation_encoder_trial_v1', 'status': 'FROZEN_NOT_LAUNCHED',
        'hypothesis': 'Knee-pretrained encoder improves fixed global-token study classification',
        'candidate': 'OrthoFoundation-L; released author forward without RoPE',
        'features': 'CLS1024 + mean196patch1024; exclude four register tokens; 20 slice tokens',
        'preprocessing': 'fixed owned wide cache; deterministic 4 centers x5 slots; 224 bilinear RGB; ImageNet normalization',
        'inference_chunk_frames': 16, 'feature_dtype': 'float16',
        'control': 'Pinned GSP02 frozen RadImageNet global features',
        'labels': 'Unchanged four-source median clipped .01-.99; weight clip(1-range,.25,1)',
        'training_rows': 4345, 'expert_rows': 58, 'excluded_report_related_rows': 4,
        'seeds': [20261008, 20261009, 20261010], 'epochs': 30,
        'optimizer': {'name': 'AdamW', 'lr': .0003, 'weight_decay': .01, 'batch_size': 64, 'clip_grad_norm': 3.},
        'selection': 'Final epoch only, all three seeds, six complete paired fits; no target or weight search',
        'allocated_gpu_hours_cap': 1.5, 'reserve_gpu_hours': 2,
        'worker_seconds_cap': 3300, 'parent_seconds_cap': 3600,
        'resource_pilot': 'First256 feature rows; head pilot on weak-only subset; full recipe projected with20% margin',
        'fixture_limits': {'cpu_gpu_fp32_max_abs_error': .003, 'gpu_amp_fp32_max_abs_error': .05},
        'screening_gates': {'candidate_macro_min': .90, 'mean_paired_macro_gain_min': .003,
                            'positive_seeds_min': 2, 'paired_bootstrap_lower_95_min_exclusive': 0},
        'bootstrap': {'seed': 20261010, 'samples': 2000, 'unit': 'study, paired across arms and seeds'},
        'limitations': ['Gold58 reused; not a fresh unbiased validation cohort',
                       'Four weak sources are correlated and partly expert-informed',
                       'Report similarity exclusions do not establish patient independence',
                       'Upstream pretraining overlap with competition patients is unknown',
                       'Control is a frozen RadImageNet head, not the0.944 B03 ensemble',
                       'A passing screen does not authorize a leaderboard score forecast'],
        'competition_submission_enabled': False,
    }
    trial_path.parent.mkdir(exist_ok=True)
    base = json.loads((ROOT / 'runs/grade_supervision_probe_source_v2.json').read_text())
    helper = ROOT / 'training/grade_supervision_probe_v2.py'
    if sha(helper) != base['sources']['training/grade_supervision_probe_v2.py']:
        raise RuntimeError('verified support source changed')
    support = helper.read_text().split('class Encoder(nn.Module):')[0]
    support = support.replace('from grade_aware_supervision import adjusted_probability\n', '')
    support = support.replace('from torchvision.models import resnet50\n', '')
    support = support.replace("OUT=Path('/kaggle/working/grade_probe')", "OUT=Path('/kaggle/working/ortho_probe')")
    previous = json.loads((ROOT / 'runs/grade_supervision_probe_outputs_v2/grade_probe/manifest.json').read_text())
    if previous['status'] != 'COMPLETE':
        raise RuntimeError('control features incomplete')
    source_paths = [ROOT / 'training' / n for n in [
        'orthofoundation_probe.py', 'orthofoundation_fit_worker.py', 'orthofoundation_encoder.py',
        'spatial_pooling_head.py', 'e03_split_contract.py', 'e03_window_contract.py']]
    trial['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    contract = {k: base['contract'][k] for k in ['targets', 'train_sha256', 'train_ids_sha256', 'label_sources', 'cache', 'seeds']}
    contract.update(schema='orthofoundation_probe_contract_v1', trial=trial, epochs=30,
                    previous_features={n: previous['artifacts'][n] for n in ['features.npy', 'study_ids.npy', 'token_masks.npy']},
                    assets=package['files'], competition_submission_enabled=False)
    modules = {p.name: p.read_text() for p in source_paths}
    modules['ortho_probe_support.py'] = support
    modules['orthofoundation_probe.py'] = modules['orthofoundation_probe.py'].replace('# __PROBE_CONTRACT__', 'CONTRACT = ' + repr(contract))
    for name, code in modules.items():
        compile(code, name, 'exec')
    directory = ROOT / 'public_kernels/orthofoundation_probe'; directory.mkdir(exist_ok=True)
    slug = 'rsna-knee-orthofoundation-probe'
    wrapper = '''import os, signal, subprocess, sys
from pathlib import Path
MODULES = ''' + repr(modules) + '''
root=Path('/kaggle/working/ortho_probe_source');root.mkdir(exist_ok=True)
for name,code in MODULES.items():(root/name).write_text(code)
p=subprocess.Popen([sys.executable,str(root/'orthofoundation_probe.py')],start_new_session=True)
try:code=p.wait(timeout=3600)
except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
if code:raise RuntimeError(f'bounded encoder comparison failed; child exit{code}')
'''
    code_path = directory / (slug + '.py'); compile(wrapper, code_path.name, 'exec'); code_path.write_text(wrapper)
    metadata = json.loads((ROOT / 'public_kernels/spatial_pooling_probe_v2/kernel-metadata.json').read_text())
    metadata.update(id='adityakumarrai0401/' + slug, title='RSNA Knee OrthoFoundation Probe', code_file=code_path.name)
    metadata['dataset_sources'].remove('marwanmath/resnet-50-radimagenet-marwan')
    metadata['dataset_sources'].append('adityakumarrai0401/rsna-knee-orthofoundation-assets-v1')
    metadata_path = directory / 'kernel-metadata.json'; metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    trial_path.write_text(json.dumps(trial, indent=2) + '\n')
    gates = [trial_path, package_path, ROOT / 'runs/spatial_pooling_probe_decision_v2.json',
             ROOT / 'research/knee_pretraining/orthofoundation/compatibility_audit.json']
    local_scripts = [ROOT / 'scripts' / n for n in [
        'build_orthofoundation_probe.py', 'audit_orthofoundation_probe.py',
        'launch_orthofoundation_probe.py', 'watch_orthofoundation_probe.py']]
    build = {'status': 'BUILT_NOT_LAUNCHED', 'contract': contract,
             'sources': {str(p.relative_to(ROOT)): sha(p) for p in source_paths + [helper, code_path, metadata_path] + local_scripts},
             'gates': {str(p.relative_to(ROOT)): sha(p) for p in gates}}
    (ROOT / 'runs/orthofoundation_probe_source.json').write_text(json.dumps(build, indent=2) + '\n')
    print(json.dumps({'status': build['status'], 'kernel': metadata['id'], 'submission': False}))


if __name__ == '__main__':
    main()
