"""Package the unchanged classifier only after measured full-corpus review."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'
PACKAGE = ROOT / 'public_kernels/mars_frozen_classifier'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    approval_path = BASE / 'classifier_resource_approval.json'
    if not approval_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_MEASURED_COMPLETE_FEATURE_CORPUS_REVIEW',
                          'classifier_training_started': False, 'new_gpu_jobs': 0})); return
    approval = json.loads(approval_path.read_text())
    assert approval['decision'] == 'APPROVE_COMPLETE_FROZEN_FEATURE_CLASSIFIER'
    assert approval['complete_study_coverage'] == 4407 and approval['numerical_and_cost_review_passed']
    assert not (BASE / 'classifier_launch.json').exists(), 'Preserve any accepted or uncertain run'
    protocol_path = BASE / 'classifier_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert sha(protocol_path) == approval['classifier_protocol_sha256']
    for relative, expected in protocol['sources_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    names = ['training/mars_classifier_train.py', 'training/mars_spatial_head.py',
             'training/mars_runtime_inputs.py', 'inference/mars_classifier_driver.py',
             'inference/mars_head_inference.py', 'scripts/replay_mars_heldout_heads.py']
    modules = {name: (ROOT / name).read_text() for name in names}
    modules['classifier_protocol.json'] = protocol_path.read_text()
    modules['classifier_resource_approval.json'] = approval_path.read_text()
    modules['serving_protocol.json'] = (BASE / 'serving_protocol.json').read_text()
    contract = {'schema': 'mars_native_complete_frozen_classifier_execution_v1',
                'torch_version': approval['torch_version'],
                'maximum_parent_seconds': approval['maximum_parent_seconds'],
                'platform_timeout_seconds': approval['platform_timeout_seconds'],
                'module_sha256': {name: hashlib.sha256(value.encode()).hexdigest()
                                  for name, value in modules.items()}}
    assert 0 < contract['maximum_parent_seconds'] < contract['platform_timeout_seconds']
    modules['classifier_contract.json'] = json.dumps(contract, indent=2) + '\n'
    code = 'MODULES=' + repr(modules) + '\n' + (
        "exec(compile(MODULES['inference/mars_classifier_driver.py'], 'mars_classifier_driver.py', 'exec'))\n")
    ast.parse(code)
    old = json.loads((ROOT / 'public_kernels/mars_resource_cuda_fix/pilot.ipynb').read_text())
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': old['metadata'],
                'cells': [{'cell_type': 'code', 'id': 'classifier-driver', 'metadata': {},
                           'execution_count': None, 'outputs': [], 'source': code}]}
    metadata = json.loads((ROOT / 'public_kernels/mars_resource_cuda_fix/kernel-metadata.json').read_text())
    metadata.pop('id_no', None)
    datasets = set()
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    for receipt in frozen['fold_reconstruction_only_sources'].values():
        owner, slug = Path(receipt['path']).parent.name.split('_', 1)
        datasets.add(owner + '/' + slug)
    metadata.update(id='adityakumarrai0401/rsna-knee-mars-frozen-spatial-classifier',
                    title='RSNA Knee MARS Frozen Spatial Classifier',
                    dataset_sources=sorted(datasets),
                    kernel_sources=['adityakumarrai0401/rsna-knee-mars-full-frozen-corpus'],
                    machine_shape=approval['machine_shape'])
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE / 'pilot.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    record = {'status': 'FIXED_FIVE_FOLD_CLASSIFIER_PACKAGED_NOT_LAUNCHED',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'code_cell_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'notebook_sha256': sha(PACKAGE / 'pilot.ipynb'),
              'metadata_sha256': sha(PACKAGE / 'kernel-metadata.json'),
              'protocol_sha256': sha(protocol_path), 'resource_approval_sha256': sha(approval_path),
              'local_image_or_feature_or_label_arrays_embedded': False,
              'classifier_training_started': False, 'new_gpu_jobs': 0}
    (BASE / 'classifier_build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
