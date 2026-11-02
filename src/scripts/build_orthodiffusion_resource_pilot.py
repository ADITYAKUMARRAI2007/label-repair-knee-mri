"""Freeze and package one private synthetic resource check on free Kaggle GPU."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/orthodiffusion_resource_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT/'launch.json').exists() and not (OUT/'contract.json').exists()
    fixture = json.loads((OUT/'cpu_fixture_audit.json').read_text())
    assert fixture['status'] == 'EXACT_AUTHOR_AND_PRIOR_FIXTURE_PARITY'
    assert sha(OUT/'cpu_fixture.npz') == fixture['fixture_sha256']
    adapter = ROOT/'training/orthodiffusion_encoder.py'
    assert sha(adapter) == fixture['adapter_sha256']
    research = ROOT/'research/knee_pretraining/orthodiffusion'
    receipt = json.loads((research/'source_receipt.json').read_text())
    modules = {}
    source_files = {}
    for item in receipt['files']:
        if item['path'] == 'LICENSE' or item['path'].startswith('diffusion_model/'):
            path = research/'source'/item['path']
            assert sha(path) == item['sha256']
            source_files[item['path']] = item['sha256']
            modules['author/'+item['path']] = path.read_text()
    hub = json.loads((research/'hub_metadata.json').read_text())
    checkpoints = {}
    for item in hub['siblings']:
        if item['rfilename'].endswith('_model.pt'):
            checkpoints[item['rfilename'].split('_')[0]] = dict(file=item['rfilename'],
                sha256=item['lfs']['sha256'], bytes=item['size'])
    assert set(checkpoints) == {'axial','coronal','sagittal'}
    contract = dict(schema='orthodiffusion_synthetic_resource_v1',
        source_commit=receipt['commit'], weights_revision=hub['sha'],
        source_files=source_files, checkpoints=checkpoints,
        adapter_sha256=sha(adapter), cpu_fixture_sha256=fixture['fixture_sha256'],
        seed=20261016, timestep=30, block='mid_2', full_input_shape=[1,1,16,256,256],
        expected_feature_shape=[1,256,1,16,16],
        cpu_gpu_max_abs_error=.003, amp_max_abs_error=.05, amp_normalized_rmse=.01,
        tolerance_note='Fixed engineering tolerances: FP32 absolute .003; AMP absolute .05 and RMS-relative1%. These do not guarantee prediction-level equivalence.',
        timed_amp_forwards=8, worker_seconds_cap=600, parent_seconds_cap=660,
        allocated_gpu_hours_cap=.5, reserve_gpu_hours=2,
        gpu_memory_cap_bytes=13*2**30, competition_submission_enabled=False,
        clinical_accuracy_measured=False, labels_used=False,
        purpose='Full-size author encoder feasibility before any accuracy experiment')
    contract_path = OUT/'contract.json'
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    worker = ROOT/'training/orthodiffusion_resource_pilot.py'
    modules.update({'orthodiffusion_encoder.py':adapter.read_text(),
                    'orthodiffusion_resource_pilot.py':worker.read_text(),
                    'contract.json':contract_path.read_text()})
    for name, code in modules.items():
        if name.endswith('.py'): compile(code,name,'exec')
    encoded = base64.b64encode((OUT/'cpu_fixture.npz').read_bytes()).decode()
    wrapper = '''import base64, os, signal, subprocess, sys
from pathlib import Path
MODULES = '''+repr(modules)+'''
FIXTURE = '''+repr(encoded)+'''
root=Path('/kaggle/working/orthodiffusion_resource_source');root.mkdir(exist_ok=True)
for name,code in MODULES.items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(code)
(root/'cpu_fixture.npz').write_bytes(base64.b64decode(FIXTURE))
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
p=subprocess.Popen([sys.executable,str(root/'orthodiffusion_resource_pilot.py')],env=env,start_new_session=True)
try:code=p.wait(timeout=660)
except BaseException:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
if code:raise RuntimeError(f'bounded OrthoDiffusion resource pilot child exit {code}')
'''
    directory = ROOT/'public_kernels/orthodiffusion_resource_pilot'
    directory.mkdir(exist_ok=True)
    slug = 'rsna-knee-orthodiffusion-resource-pilot'
    code_path = directory/(slug+'.py')
    compile(wrapper,code_path.name,'exec');code_path.write_text(wrapper)
    metadata = dict(id='adityakumarrai0401/'+slug, title='RSNA Knee OrthoDiffusion Resource Pilot',
        code_file=code_path.name, language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=False, machine_shape='NvidiaTeslaT4',
        dataset_sources=['nasrinakbari/orthodiffusion-rsna-knee-prenuvo-v1'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    meta_path = directory/'kernel-metadata.json'
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    paths = [adapter,worker,code_path,meta_path,contract_path,OUT/'cpu_fixture.npz',
             OUT/'cpu_fixture_audit.json',Path(__file__),ROOT/'scripts/launch_orthodiffusion_resource_pilot.py']
    build = dict(status='BUILT_NOT_LAUNCHED',contract=contract,
                 sources={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (OUT/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps(dict(status=build['status'],kernel=metadata['id'],
                         code_bytes=code_path.stat().st_size,contract_sha256=sha(contract_path)),indent=2))

if __name__ == '__main__':main()
