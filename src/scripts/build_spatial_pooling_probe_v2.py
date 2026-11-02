#!/usr/bin/env python3
"""Build the frozen spatial comparison with the verified input/resource helpers."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    trial_path=ROOT/'runs/spatial_pooling_trial_v1/contract.json'
    trial=json.loads(trial_path.read_text())
    if trial['status']!='FROZEN_NOT_LAUNCHED':raise RuntimeError('trial contract not frozen')
    for relative,expected in trial['source_sha256'].items():
        if sha(ROOT/relative)!=expected:raise RuntimeError('frozen source changed: '+relative)
    repair_path=ROOT/'runs/spatial_pooling_execution_repair.json'
    repair=json.loads(repair_path.read_text())
    if repair['status']!='LOCAL_EQUIVALENCE_PASS':raise RuntimeError('execution repair not verified')
    for relative,expected in repair['source_sha256'].items():
        if sha(ROOT/relative)!=expected:raise RuntimeError('repair source changed: '+relative)
    base=json.loads((ROOT/'runs/grade_supervision_probe_source_v2.json').read_text())
    helper_path=ROOT/'training/grade_supervision_probe_v2.py'
    if sha(helper_path)!=base['sources']['training/grade_supervision_probe_v2.py']:
        raise RuntimeError('verified helper source changed')
    text=helper_path.read_text()
    support=text[:text.index('class Head(nn.Module):')]
    support=support.replace('from grade_aware_supervision import adjusted_probability\n','')
    support=support.replace("OUT=Path('/kaggle/working/grade_probe')","OUT=Path('/kaggle/working/spatial_probe')")
    old='def forward(self,x):return self.backbone(x).mean((2,3))'
    if support.count(old)!=1:raise RuntimeError('encoder structure changed')
    support=support.replace(old,'def forward(self,x):return self.backbone(x)')
    previous=json.loads((ROOT/'runs/grade_supervision_probe_outputs_v2/grade_probe/manifest.json').read_text())
    if previous['status']!='COMPLETE':raise RuntimeError('prior features incomplete')
    contract={key:base['contract'][key] for key in ['targets','train_sha256','train_ids_sha256','encoder_sha256','label_sources','cache','seeds']}
    contract.update(schema='spatial_pooling_probe_contract_v2',execution_repair=repair,trial=trial,trial_contract_sha256=sha(trial_path),
                    previous_features={name:previous['artifacts'][name] for name in ['features.npy','token_masks.npy','study_ids.npy']},
                    epochs=30,competition_submission_enabled=False,
                    expert_data_limitations='Gold58 reused; four-file control is correlated and partly Gold-informed; no patient-independent validation claim')
    worker=ROOT/'training/spatial_pooling_probe_v2.py';head=ROOT/'training/spatial_pooling_head.py'
    other=[ROOT/'training/spatial_fit_worker.py',ROOT/'training/e03_split_contract.py',ROOT/'training/e03_window_contract.py']
    modules={p.name:p.read_text() for p in [head]+other}
    modules['spatial_probe_support.py']=support
    modules[worker.name]=worker.read_text().replace('# __PROBE_CONTRACT__','CONTRACT = '+repr(contract))
    for name,code in modules.items():compile(code,name,'exec')
    directory=ROOT/'public_kernels/spatial_pooling_probe_v2';directory.mkdir(exist_ok=True)
    if (ROOT/'runs/spatial_pooling_probe_launch_v2.json').exists():raise RuntimeError('preserve launched source')
    slug='rsna-knee-spatial-pooling-probe-v2';code_path=directory/(slug+'.py')
    wrapper='''import os, signal, subprocess, sys
from pathlib import Path
MODULES = '''+repr(modules)+'''
root=Path('/kaggle/working/spatial_probe_source');root.mkdir(exist_ok=True)
for name,code in MODULES.items():(root/name).write_text(code)
p=subprocess.Popen([sys.executable,str(root/'spatial_pooling_probe_v2.py')],start_new_session=True)
try:code=p.wait(timeout=3600)
except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
if code:raise RuntimeError(f'bounded spatial comparison failed; child exit{code}')
'''
    compile(wrapper,code_path.name,'exec');code_path.write_text(wrapper)
    metadata={'id':'adityakumarrai0401/'+slug,'title':'RSNA Knee Spatial Pooling Probe V2',
              'code_file':code_path.name,'language':'python','kernel_type':'script','is_private':True,
              'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
              'dataset_sources':['marwanmath/resnet-50-radimagenet-marwan','pilkwang/rsna-knee-llm-labels',
                  'lixin73/rsna-knee-llm-report-labels-sol56','stevenleehans/rsna-knee-llm-report-labels','dreaddevelopment/rsna-knee-labels'],
              'competition_sources':['rsna-knee-abnormality-detection'],
              'kernel_sources':['adityakumarrai0401/rsna-knee-e04-wide-cache-base',
                  'adityakumarrai0401/rsna-knee-e04-wide-cache-extension','adityakumarrai0401/rsna-knee-grade-supervision-probe-v2'],
              'model_sources':[]}
    metadata_path=directory/'kernel-metadata.json';metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    source_paths=[worker,head,helper_path]+other+[code_path,metadata_path]
    gates=[trial_path,repair_path,ROOT/'runs/spatial_pooling_probe_decision.json',ROOT/'runs/spatial_head_local_check.json',ROOT/'runs/public0033_probe_decision.json']
    build={'status':'BUILT_NOT_LAUNCHED','contract':contract,
           'sources':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
           'gates':{str(p.relative_to(ROOT)):sha(p) for p in gates}}
    (ROOT/'runs/spatial_pooling_probe_source_v2.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps({'status':build['status'],'kernel':metadata['id'],'gpu_cap_hours':1.5,'spatial_cells':49,'submission':False}))


if __name__=='__main__':main()
