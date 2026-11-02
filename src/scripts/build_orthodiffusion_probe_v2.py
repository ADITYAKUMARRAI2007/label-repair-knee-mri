"""Freeze a single complete 3D representation comparison and its resource gates."""
import base64
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/orthodiffusion_trial_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT/'launch.json').exists() and not (OUT/'contract.json').exists()
    head=json.loads((ROOT/'runs/orthodiffusion_trial_v1/head_check.json').read_text());assert head['status']=='PASS_HEAD_MASK_UPDATE_REPLAY'
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
    support=support.replace("OUT=Path('/kaggle/working/grade_probe')","OUT=Path('/kaggle/working/diffusion_probe_v2')")
    prior=json.loads((ROOT/'runs/grade_supervision_probe_outputs_v2/grade_probe/manifest.json').read_text())
    pilot=json.loads((ROOT/'runs/orthodiffusion_resource_v1/contract.json').read_text())
    original_path=ROOT/'runs/orthodiffusion_trial_v1/contract.json'
    original_build=json.loads((ROOT/'runs/orthodiffusion_trial_v1/build.json').read_text())
    assert sha(original_path)==original_build['sources'][str(original_path.relative_to(ROOT))]
    for name in ['orthodiffusion_fit_worker.py','orthodiffusion_head.py','orthodiffusion_encoder.py',
                 'spatial_pooling_head.py','e03_split_contract.py']:
        path=ROOT/'training'/name
        assert sha(path)==original_build['sources'][str(path.relative_to(ROOT))]
    contract=json.loads(original_path.read_text())
    repair_path=ROOT/'runs/orthodiffusion_volume_v2/repair_audit.json'
    repair=json.loads(repair_path.read_text())
    assert repair['status']=='PASS_SHORT_VOLUME_REPAIR_AND_UNCHANGED_LONG_VOLUME_PIXELS'
    assert sha(ROOT/'inference/orthodiffusion_volume_v2.py')==repair['reader_sha256']
    recovery_path=OUT/'checkpoint_recovery_check.json'
    recovery=json.loads(recovery_path.read_text())
    assert recovery['status']=='PASS_INTERRUPTED_WRITE_RECOVERY_AND_CORRUPTION_REJECTION'
    assert sha(ROOT/'training/feature_checkpoint.py')==recovery['source_sha256']
    listings_path=ROOT/'runs/orthodiffusion_volume_v2/weak_short_study_listings.json'
    listings=json.loads(listings_path.read_text())
    short_ids=sorted({r['metadata']['StudyInstanceUID'] for r in listings})
    assert len(short_ids)==1 and all(len(r['listing']['files'])==15 for r in listings)
    contract.update(schema='orthodiffusion_three_plane_trial_v2',
        original_volume_sha256=contract['volume_sha256'],volume_sha256=repair['reader_sha256'],
        volume_policy='Original strict native-axis reader; central16 if available; otherwise symmetric edge replication to16; whole-field256linear; global minmax[-1,1]',
        plane_policy='Preserve original >=16-series selections; otherwise highest-ranked valid short series; explicit missing-plane mask',
        checkpoint_policy='Persistent feature array with fsync plus atomic completion journal every64 rows, every32 during preflight, and ordinary exit; row SHA256 validates recovery',
        checkpoint_helper_sha256=recovery['source_sha256'])
    contract['preflight'].update(studies=129,additional_weak_short_studies=short_ids,
        selection='Original128 sha256(20261017:StudyInstanceUID) weak-only cases plus specified weak-only short-volume case')
    contract['runtime_policy']='129 real weak-only decode+extraction including short-volume case; two concurrent50step head pilots(10warmup); full recipe projected with20% margin; no reduced schedule'
    feature_keys=['source_files','source_commit','checkpoints','weights_revision','adapter_sha256',
        'original_volume_sha256','volume_sha256','train_series_sha256','planes','feature_shape',
        'feature_dtype','timestep','block','noise_seed_base','volume_policy','plane_policy']
    feature_contract={k:contract[k] for k in feature_keys}
    feature_contract['study_ids_sha256']=contract['previous_features']['study_ids.npy']['sha256']
    contract['feature_contract']=feature_contract
    contract['feature_contract_sha256']=hashlib.sha256(json.dumps(feature_contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    contract_path=OUT/'contract.json';contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    paths=[ROOT/'training'/n for n in ['orthodiffusion_probe_v2.py','feature_checkpoint.py','orthodiffusion_fit_worker.py','orthodiffusion_head.py',
                                     'orthodiffusion_encoder.py','spatial_pooling_head.py','e03_split_contract.py']]
    paths += [ROOT/'inference'/n for n in ['orthodiffusion_volume.py','orthodiffusion_volume_v2.py','orthodiffusion_decode_worker_v2.py']]
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
p=subprocess.Popen([sys.executable,str(root/'orthodiffusion_probe_v2.py')],env=env,start_new_session=True)
try:code=p.wait(timeout=6300)
except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
if code:raise RuntimeError(f'bounded OrthoDiffusion trial child exit {code}')
'''
    directory=ROOT/'public_kernels/orthodiffusion_probe_v2';directory.mkdir(exist_ok=True)
    slug='rsna-knee-orthodiffusion-probe-v2';code_path=directory/(slug+'.py')
    compile(wrapper,code_path.name,'exec');code_path.write_text(wrapper)
    metadata=dict(id='adityakumarrai0401/'+slug,title='RSNA Knee OrthoDiffusion Probe V2',code_file=code_path.name,
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        machine_shape='NvidiaTeslaT4',dataset_sources=['pilkwang/rsna-knee-llm-labels','lixin73/rsna-knee-llm-report-labels-sol56',
        'stevenleehans/rsna-knee-llm-report-labels','dreaddevelopment/rsna-knee-labels','nasrinakbari/orthodiffusion-rsna-knee-prenuvo-v1'],
        competition_sources=['rsna-knee-abnormality-detection'],
        kernel_sources=['adityakumarrai0401/rsna-knee-grade-supervision-probe-v2'],model_sources=[])
    meta_path=directory/'kernel-metadata.json';meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    paths += [helper,code_path,meta_path,contract_path,fixture,Path(__file__),ROOT/'scripts/launch_orthodiffusion_probe_v2.py',ROOT/'scripts/audit_orthodiffusion_probe_v2.py']
    gates=[ROOT/'runs/orthodiffusion_trial_v1/head_check.json',repair_path,recovery_path,listings_path,ROOT/'runs/orthodiffusion_volume_v1/synthetic_audit.json',
           ROOT/'runs/orthodiffusion_resource_v1/independent_audit.json',ROOT/'runs/orthodiffusion_resource_v1/decision.json']
    build=dict(status='BUILT_NOT_LAUNCHED',contract=contract,
               sources={str(p.relative_to(ROOT)):sha(p) for p in paths},gates={str(p.relative_to(ROOT)):sha(p) for p in gates})
    (OUT/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps(dict(status=build['status'],kernel=metadata['id'],source_bytes=code_path.stat().st_size,
                         competition_submission_enabled=False),indent=2))

if __name__=='__main__':main()
