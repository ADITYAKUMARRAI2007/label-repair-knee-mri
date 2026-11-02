"""Build one offline GPU resource pilot, without uploading or launching."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
PACKAGE = ROOT / 'public_kernels/mars_resource_pilot_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


DRIVER = '''
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
work=Path('/kaggle/working')
root=work/'mars_pilot_source';root.mkdir(exist_ok=False)
for name,source in MODULES.items():
    (root/name).write_text(source)
contract=json.loads(MODULES['pilot_contract.json'])
started=time.monotonic()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def remaining():
    value=contract['bootstrap_and_worker_seconds_cap']-(time.monotonic()-started)
    if value<=0:raise TimeoutError('Fixed resource pilot budget exhausted')
    return value
def attached(slug,filename):
    return next(p for p in [Path('/kaggle/input')/slug,
                            Path('/kaggle/input/datasets/adityakumarrai0401')/slug]
                if (p/filename).is_file())
try:
    assert sys.version_info[:2]==(3,13)
    import torch
    assert str(torch.__version__)==contract['torch_version']
    assert torch.cuda.device_count()==2
    for name,expected in contract['module_sha256'].items():assert sha(root/name)==expected,name
    assets=attached('rsna-knee-mars-model-assets-v2','MARS.pt')
    for name,expected in contract['asset_sha256'].items():assert sha(assets/name)==expected,name
    decoder=attached('rsna-knee-decoder-cp313','decoder_identity.json')
    wheels=[]
    for name,expected in contract['decoder_wheels'].items():
        assert sha(decoder/name)==expected,name
        wheels.append(str(decoder/name))
    target=work/'mars_pilot_dependencies'
    with (work/'mars_resource_bootstrap.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                        '--target',str(target),str(assets/contract['monai_wheel']),*wheels],
                       check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(180,remaining()))
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(target),str(root),os.environ.get('PYTHONPATH','')]))
        subprocess.run([sys.executable,str(root/'check_decoders.py')],env=env,
                       check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(60,remaining()))
    assert 'DECODER_ABI_PASS cp313' in (work/'mars_resource_bootstrap.log').read_text()
    data=next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                          Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
              if (p/'train_series.csv').is_file())
    with (work/'mars_resource_worker.log').open('w') as log:
        subprocess.run([sys.executable,str(root/'mars_resource_pilot.py'),
                        '--input',str(data),'--assets',str(assets),'--out',str(work/'mars_resource_pilot'),
                        '--protocol',str(root/'resource_protocol.json'),'--device','cuda:0'],
                       env=env,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=remaining())
    result=json.loads((work/'mars_resource_pilot/receipt.json').read_text())
    assert result['status']=='REAL_DICOM_CUDA_RESOURCE_PILOT_COMPLETE_NO_ACCURACY'
    assert not (work/'submission.csv').exists()
    (work/'mars_resource_execution.json').write_text(json.dumps({
        'status':'BOUNDED_PILOT_EXECUTION_COMPLETE','contract_sha256':sha(root/'pilot_contract.json'),
        'pilot_receipt_sha256':sha(work/'mars_resource_pilot/receipt.json'),
        'elapsed_seconds':time.monotonic()-started,'classification_fitting_enabled':False,
        'competition_submission_enabled':False},indent=2)+'\\n')
    print(json.dumps({'status':result['status'],'completed_studies':result['completed_studies']}),flush=True)
except BaseException as error:
    (work/'mars_resource_bootstrap_failure.json').write_text(json.dumps({
        'status':'PILOT_EXECUTION_REJECTED','error_type':type(error).__name__,
        'classification_fitting_enabled':False,'competition_submission_enabled':False})+'\\n')
    raise
'''


def main():
    assert not (RUN / 'resource_launch.json').exists(), 'Preserve any accepted or uncertain launch'
    asset_package = json.loads((RUN / 'model_asset_package.json').read_text())
    assert asset_package['status'] == 'READY_PRIVATE_PUBLIC_MODEL_ONLY_UPLOAD'
    assert asset_package['private_images_or_patient_features_included'] is False
    assets = ROOT / 'datasets/mars_model_assets_v2'
    for name, record in asset_package['files'].items():
        assert sha(assets / name) == record['sha256'], name
    preparation = json.loads((RUN / 'resource_source_preparation_v2.json').read_text())
    for name, expected in preparation['sources_sha256'].items():
        assert sha(ROOT / name) == expected, name
    protocol = json.loads((RUN / 'resource_protocol_v2.json').read_text())
    assert protocol['schema'] == 'mars_rsna_resource_pilot_v2_model_only'
    assert len(protocol['studies']) == 24 and not protocol['classification_fitting_enabled']
    source_names = ['mars_encoder.py', 'mars_legacy_merging.py', 'mars_feature_cache.py',
                    'mars_volume.py', 'mars_study.py', 'orthodiffusion_volume.py', 'mars_resource_pilot.py']
    modules = {name: (ROOT / 'inference' / name).read_text() for name in source_names}
    baseline = ROOT / 'runs/published_tri_reader/output/tri_reader_source'
    decoder_source = (baseline / 'check_decoders.py').read_text()
    assert decoder_source.count("TARGET='/kaggle/working/reader_decoder_env'") == 1
    modules['check_decoders.py'] = decoder_source.replace(
        "TARGET='/kaggle/working/reader_decoder_env'",
        "TARGET='/kaggle/working/mars_pilot_dependencies'")
    modules['resource_protocol.json'] = (RUN / 'resource_protocol_v2.json').read_text()
    baseline_contract = json.loads((baseline / 'contract.json').read_text())
    contract = {
        'schema': 'mars_resource_pilot_execution_v2_model_only',
        'module_sha256': {name: hashlib.sha256(code.encode()).hexdigest() for name, code in modules.items()},
        'asset_sha256': {name: r['sha256'] for name, r in asset_package['files'].items()
                         if name != 'dataset-metadata.json'},
        'decoder_wheels': baseline_contract['decoder_wheels']['cp313'],
        'torch_version': baseline_contract['torch_version'],
        'monai_wheel': 'monai-1.5.1-py3-none-any.whl',
        'bootstrap_and_worker_seconds_cap': 1200, 'platform_timeout_seconds': 1260,
        'classification_fitting_enabled': False, 'competition_submission_enabled': False,
        'launch_dependency': protocol['launch_dependency'],
    }
    modules['pilot_contract.json'] = json.dumps(contract, indent=2) + '\n'
    for name, code in modules.items():
        if name.endswith('.py'):
            ast.parse(code, filename=name)
    code = 'MODULES=' + repr(modules) + '\n' + DRIVER
    ast.parse(code)
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE / 'pilot.py').write_text(code)
    metadata = json.loads((ROOT / 'public_kernels/coherent_anchor_pair/kernel-metadata.json').read_text())
    metadata.update(id='adityakumarrai0401/rsna-knee-mars-resource-pilot-v2',
                    title='RSNA Knee MARS Resource Pilot V2', code_file='pilot.py',
                    dataset_sources=['adityakumarrai0401/rsna-knee-mars-model-assets-v2',
                                     'adityakumarrai0401/rsna-knee-decoder-cp313'])
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    record = {
        'status': 'PRIVATE_RESOURCE_PILOT_PACKAGED_NOT_UPLOADED_OR_LAUNCHED',
        'source_sha256': sha(PACKAGE / 'pilot.py'),
        'metadata_sha256': sha(PACKAGE / 'kernel-metadata.json'),
        'asset_package_sha256': sha(RUN / 'model_asset_package.json'),
        'resource_protocol_sha256': sha(RUN / 'resource_protocol_v2.json'),
        'modules': len(modules), 'new_gpu_jobs': 0, 'new_competition_submissions': 0,
        'gpu_execution_verified': False, 'actual_resource_timing_measured': False,
    }
    (RUN / 'resource_build_v2.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
