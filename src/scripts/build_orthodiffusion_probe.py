"""Freeze a single complete 3D representation comparison and its resource gates."""
import base64
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/orthodiffusion_trial_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT/'launch.json').exists() and not (OUT/'contract.json').exists()
    head=json.loads((OUT/'head_check.json').read_text());assert head['status']=='PASS_HEAD_MASK_UPDATE_REPLAY'
    assert sha(ROOT/'training/orthodiffusion_head.py')==head['source_sha256']
    volume=json.loads((ROOT/'runs/orthodiffusion_volume_v1/synthetic_audit.json').read_text())
    assert volume['status']=='PASS_SYNTHETIC_DICOM_AND_AUTHOR_ARRAY_TRANSFORM_ONLY'
    assert sha(ROOT/'inference/orthodiffusion_volume.py')==volume['reader_sha256']
    gpu=json.loads((ROOT/'runs/orthodiffusion_resource_v1/independent_audit.json').read_text())
    assert gpu['status']=='PASS_SAVED_NUMERICS_AND_PROJECTION_AUDIT_ONLY'
    old=json.loads((ROOT/'runs/grade_supervision_probe_source_v2.json').read_text())
    helper=ROOT/'training/grade_supervision_probe_v2.py'
    assert sha(helper)==old['sources']['training/grade_supervision_probe_v2.py']
    support=helper.read_text().split('class Encoder(nn.Module):')[0]
    for line in ['from grade_aware_supervision import adjusted_probability\n','from torchvision.models import resnet50\n',
                 'from e03_window_contract import select_slot_windows\n']:
        support=support.replace(line,'')
    support=support.replace('DEADLINE=START+3300','DEADLINE=START+6000')
    support=support.replace("OUT=Path('/kaggle/working/grade_probe')","OUT=Path('/kaggle/working/diffusion_probe')")
    prior=json.loads((ROOT/'runs/grade_supervision_probe_outputs_v2/grade_probe/manifest.json').read_text())
    pilot=json.loads((ROOT/'runs/orthodiffusion_resource_v1/contract.json').read_text())
    contract={k:old['contract'][k] for k in ['targets','train_sha256','train_ids_sha256','label_sources','seeds']}
    contract.update(schema='orthodiffusion_three_plane_trial_v1',
        hypothesis='Frozen knee-specific3D diffusion features with spatial/global pooling improve study classification over the fixed RadImageNet control',
        source_files=pilot['source_files'],source_commit=pilot['source_commit'],checkpoints=pilot['checkpoints'],
        weights_revision=pilot['weights_revision'],adapter_sha256=pilot['adapter_sha256'],
        cpu_fixture_sha256=pilot['cpu_fixture_sha256'],volume_sha256=volume['reader_sha256'],
        head_sha256=head['source_sha256'],train_series_sha256=sha(ROOT/'data/train_series.csv'),
        previous_features={n:prior['artifacts'][n] for n in ['features.npy','study_ids.npy','token_masks.npy']},
        planes=['Axial','Coronal','Sagittal'],feature_shape=[4407,3,256,256],feature_dtype='float16',
        timestep=30,block='mid_2',noise_seed_base=20261016,
        volume_policy='Shared strict DICOM reader; contiguouscentral16; whole-field256linear; global minmax[-1,1]; native acquisition axes',
        plane_policy='Fluid-sensitive then fat-suppressed preference; UID tie-break; skip fewer-than16 native slices; explicit missing-plane mask',
        candidate_head='Separate per-plane LN256/Linear256/GELU; plane embedding; fixed2D sinusoidal coordinates;12query spatial attention plus global mean; dropout0.1;12linear outputs',
        candidate_trainable_parameters=head['trainable_parameters'],
        control_head='Unchanged SpatialHead on pinned RadImageNet20globaltokens',
        comparison_scope='Full representation/head comparison; architectures differ, no parameter-matched causal claim',
        training_rows=4345,expert_rows=58,excluded_report_related_rows=4,epochs=30,
        labels='Unchanged four-source median clipped.01-.99; weight clip(1-range,.25,1)',
        optimizer=dict(name='AdamW',lr=.0003,weight_decay=.01,batch_size=64,clip_grad_norm=3),
        selection='Final epoch only; all3seeds and6fits before scoring; no target/weight/epoch search',
        preflight=dict(studies=128,selection='sha256(20261017:StudyInstanceUID) sorted, weak-only',
                       decode_failures_allowed=0,replay_studies=3,replay_max_error=0),
        worker_seconds_cap=6000,parent_seconds_cap=6300,allocated_gpu_hours_cap=2.,reserve_gpu_hours=2.,
        runtime_policy='128 real weak-only decode+extraction; two concurrent50step head pilots(10warmup); full recipe projected with20% margin; no reduced schedule',
        screening_gates=dict(candidate_macro_min=.90,mean_paired_macro_gain_min=.003,positive_seeds_min=2,
                             paired_bootstrap_lower_95_min_exclusive=0),
        bootstrap=dict(seed=20261010,samples=2000,unit='study; paired across arms, seeds and targets'),
        competition_submission_enabled=False,
        limitations=['ReusedGold58 is a rejection screen, not fresh independent validation',
                     'Weak sources are correlated and partly expert-informed; patient independence not established',
                     'Pretraining patient overlap unknown; author upstream DICOM-to-NIfTI orientation not available',
                     'Control is a frozen RadImageNet head, not the0.944 B03 ensemble',
                     'Passing this comparison does not establish a leaderboard gain'])
    OUT.mkdir(exist_ok=True)
    contract_path=OUT/'contract.json';contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    paths=[ROOT/'training'/n for n in ['orthodiffusion_probe.py','orthodiffusion_fit_worker.py','orthodiffusion_head.py',
                                     'orthodiffusion_encoder.py','spatial_pooling_head.py','e03_split_contract.py']]
    paths += [ROOT/'inference'/n for n in ['orthodiffusion_volume.py','orthodiffusion_decode_worker.py']]
    modules={p.name:p.read_text() for p in paths}
    modules.update({'diffusion_probe_support.py':support,'contract.json':contract_path.read_text()})
    research=ROOT/'research/knee_pretraining/orthodiffusion/source'
    for name,expected in contract['source_files'].items():
        assert sha(research/name)==expected
        modules['author/'+name]=(research/name).read_text()
    for name,code in modules.items():
        if name.endswith('.py'):compile(code,name,'exec')
    fixture=ROOT/'runs/orthodiffusion_resource_v1/cpu_fixture.npz'
    assert sha(fixture)==contract['cpu_fixture_sha256']
    encoded=base64.b64encode(fixture.read_bytes()).decode()
    wrapper='''import base64,os,signal,subprocess,sys
from pathlib import Path
MODULES = '''+repr(modules)+'''
FIXTURE = '''+repr(encoded)+'''
root=Path('/kaggle/working/diffusion_probe_source');root.mkdir(exist_ok=True)
for name,code in MODULES.items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(code)
(root/'cpu_fixture.npz').write_bytes(base64.b64decode(FIXTURE))
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
p=subprocess.Popen([sys.executable,str(root/'orthodiffusion_probe.py')],env=env,start_new_session=True)
try:code=p.wait(timeout=6300)
except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
if code:raise RuntimeError(f'bounded OrthoDiffusion trial child exit {code}')
'''
    directory=ROOT/'public_kernels/orthodiffusion_probe_v1';directory.mkdir(exist_ok=True)
    slug='rsna-knee-orthodiffusion-probe-v1';code_path=directory/(slug+'.py')
    compile(wrapper,code_path.name,'exec');code_path.write_text(wrapper)
    metadata=dict(id='adityakumarrai0401/'+slug,title='RSNA Knee OrthoDiffusion Probe V1',code_file=code_path.name,
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        machine_shape='NvidiaTeslaT4',dataset_sources=['pilkwang/rsna-knee-llm-labels','lixin73/rsna-knee-llm-report-labels-sol56',
        'stevenleehans/rsna-knee-llm-report-labels','dreaddevelopment/rsna-knee-labels','nasrinakbari/orthodiffusion-rsna-knee-prenuvo-v1'],
        competition_sources=['rsna-knee-abnormality-detection'],
        kernel_sources=['adityakumarrai0401/rsna-knee-grade-supervision-probe-v2'],model_sources=[])
    meta_path=directory/'kernel-metadata.json';meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    paths += [helper,code_path,meta_path,contract_path,fixture,Path(__file__),ROOT/'scripts/launch_orthodiffusion_probe.py']
    gates=[OUT/'head_check.json',ROOT/'runs/orthodiffusion_volume_v1/synthetic_audit.json',
           ROOT/'runs/orthodiffusion_resource_v1/independent_audit.json',ROOT/'runs/orthodiffusion_resource_v1/decision.json']
    build=dict(status='BUILT_NOT_LAUNCHED',contract=contract,
               sources={str(p.relative_to(ROOT)):sha(p) for p in paths},gates={str(p.relative_to(ROOT)):sha(p) for p in gates})
    (OUT/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps(dict(status=build['status'],kernel=metadata['id'],source_bytes=code_path.stat().st_size,
                         competition_submission_enabled=False),indent=2))

if __name__=='__main__':main()
