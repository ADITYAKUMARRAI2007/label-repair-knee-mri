#!/usr/bin/env python3
"""Build a private bounded paired probe using only hosted competition inputs."""
import hashlib
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,choices=[1,2],default=1)
    attempt=parser.parse_args().attempt
    suffix='' if attempt==1 else '_v2'
    repair_path=ROOT/'runs/grade_supervision_probe_memory_repair.json'
    if attempt==2:
        if json.loads(repair_path.read_text())['status']!='PASS_PLATFORM_REPAIR_CHECKS':
            raise RuntimeError('platform repair check missing')
    trial=json.loads((ROOT/'runs/grade_supervision_trial_v1/contract.json').read_text())
    audit_path=ROOT/'runs/grade_supervision_trial_v1/label_audit.json'
    audit=json.loads(audit_path.read_text())
    if trial['status']!='FROZEN_NOT_TRAINED' or audit['status']!='PASS_NARROW_TEXT_AND_ARRAY_CHECKS':
        raise RuntimeError('supervision audit gate missing')
    if sha(ROOT/'training/grade_aware_supervision.py')!=trial['policy_source_sha256']:
        raise RuntimeError('frozen policy changed')
    for name,expected in trial['artifacts'].items():
        if sha(ROOT/'runs/grade_supervision_trial_v1'/name)!=expected:raise RuntimeError('frozen trial artifact changed')
    review=json.loads((ROOT/'runs/supervision_review_v1/split.json').read_text())
    split=json.loads((ROOT/'runs/E03_contract/split.json').read_text())
    excluded=set(review['excluded_expert_and_connected_ids'])
    if excluded!=set(split['gold'])|set(split['excluded_gold_group_weak']):raise RuntimeError('report exclusions changed')
    frozen=json.loads((ROOT/'runs/E04_contract/e04_contract.json').read_text())
    train_ids=sorted(r['StudyInstanceUID'] for r in frozen['rows'] if r['StudyInstanceUID'] not in excluded)
    contract={'schema':'grade_supervision_probe_contract_v1','targets':trial['target_order'],
              'train_sha256':frozen['official_train_sha256'],
              'train_ids_sha256':hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),
              'encoder_sha256':'08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734',
              'label_sources':{name:r['sha256'] for name,r in frozen['fusion']['source_receipts'].items()},
              'cache':{part:sha(ROOT/f'runs/E04_wide_cache_{part}/e04_wide_{part}_manifest.json') for part in ['base','extension']},
              'changed_cells':trial['changed_cells'],'seeds':[20261008,20261009,20261010],
              'epochs':30,'checkpoint_rule':'final_epoch30_only','feature_slots':5,'slices_per_slot':4,
              'feature_representation':'owned6-94pct140mm44slice; four deterministic slot centers; grayscaleRGB; bilinear224; [-1,1]',
              'comparison_rule':'preserve anchor unless candidate mean-seed macro>=.90 and candidate-control macro>=.003; targeted mean-seed AUC gain>=.01; positive macro gain in2of3 paired seeds; paired study-bootstrap lower95pct bound>0; otherwise reject or inconclusive',
              'expert_data_limitations':'Gold58 reused; correlated label sources include Gold-informed priors; no pristine clinical validation claim',
              'gpu_budget_hours_including_startup_margin':1.5,'child_max_seconds':3600,
              'competition_submission_enabled':False,'trial_contract_sha256':sha(ROOT/'runs/grade_supervision_trial_v1/contract.json'),
              'label_audit_sha256':sha(audit_path)}
    paths=[ROOT/'training/grade_aware_supervision.py',ROOT/'training/e03_split_contract.py',
           ROOT/'training/e03_window_contract.py',ROOT/f'training/grade_supervision_probe{suffix}.py']
    modules={p.name:p.read_text() for p in paths}
    if attempt==2:
        modules['grade_supervision_probe.py']=modules.pop('grade_supervision_probe_v2.py')
    modules['grade_supervision_probe.py']=modules['grade_supervision_probe.py'].replace('# __PROBE_CONTRACT__','CONTRACT = '+repr(contract))
    for name,source in modules.items():compile(source,name,'exec')
    slug=f'rsna-knee-grade-supervision-probe-v{attempt}'
    directory=ROOT/f'public_kernels/grade_supervision_probe_v{attempt}';directory.mkdir(parents=True,exist_ok=True)
    launch=ROOT/f'runs/grade_supervision_probe_launch{suffix}.json'
    if launch.exists():raise RuntimeError('preserve already launched source')
    source='''import json, os, signal, subprocess, sys\nfrom pathlib import Path\nMODULES = '''+repr(modules)+'''\nroot=Path('/kaggle/working/grade_probe_source');root.mkdir(exist_ok=True)\nfor name,source in MODULES.items():(root/name).write_text(source)\np=subprocess.Popen([sys.executable,str(root/'grade_supervision_probe.py')],start_new_session=True)\ntry:\n    code=p.wait(timeout=3600)\nexcept BaseException:\n    os.killpg(p.pid,signal.SIGTERM)\n    try:p.wait(timeout=10)\n    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()\n    raise\nif code:raise RuntimeError(f'bounded comparison failed; child exit {code}')\n'''
    compile(source,slug+'.py','exec')
    code_path=directory/(slug+'.py');code_path.write_text(source)
    metadata={'id':'adityakumarrai0401/'+slug,'title':f'RSNA Knee Grade Supervision Probe V{attempt}',
              'code_file':code_path.name,'language':'python','kernel_type':'script','is_private':True,
              'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
              'machine_shape':'NvidiaTeslaT4','model_sources':[],'dataset_sources':[
                  'marwanmath/resnet-50-radimagenet-marwan','pilkwang/rsna-knee-llm-labels',
                  'lixin73/rsna-knee-llm-report-labels-sol56','stevenleehans/rsna-knee-llm-report-labels',
                  'dreaddevelopment/rsna-knee-labels'],
              'competition_sources':['rsna-knee-abnormality-detection'],
              'kernel_sources':['adityakumarrai0401/rsna-knee-e04-wide-cache-base',
                                'adityakumarrai0401/rsna-knee-e04-wide-cache-extension']}
    (directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    receipt={'status':'BUILT_NOT_LAUNCHED','contract':contract,
             'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths+[code_path,directory/'kernel-metadata.json']},
             'gates':{str(p.relative_to(ROOT)):sha(p) for p in [audit_path,ROOT/'runs/grade_supervision_trial_v1/contract.json']}}
    if attempt==2:receipt['gates'][str(repair_path.relative_to(ROOT))]=sha(repair_path)
    (ROOT/f'runs/grade_supervision_probe_source{suffix}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'kernel':metadata['id'],'gpu_hours_cap':1.5,'paired_seeds':3,'submission':False}))


if __name__=='__main__':main()
