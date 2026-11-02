"""Package the diagnosed CUDA initialization repair without altering prior runs."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'
PACKAGE = ROOT / 'public_kernels/mars_resource_cuda_fix'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_path = ROOT / 'public_kernels/mars_resource_restart_notebook/pilot.ipynb'
    old = json.loads(old_path.read_text())
    code_cell = next(c for c in old['cells'] if c['cell_type'] == 'code')
    modules = ast.literal_eval(ast.parse(code_cell['source']).body[0].value)
    previous = dict(modules)
    old_worker = modules['mars_resource_pilot.py']
    new_worker = (ROOT / 'inference/mars_resource_pilot.py').read_text()
    original_line = '        torch.cuda.reset_peak_memory_stats(device)\n'
    replacement = """        stage = 'cuda_initialization'
        # CUDA availability/count queries do not initialize the allocator.
        # reset_peak_memory_stats enters the allocator directly on this runtime.
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({'stage': 'cuda_memory_counter_ready',
                          'device': str(device), 'cuda_initialized': torch.cuda.is_initialized()}),
              flush=True)
"""
    assert old_worker.count(original_line) == 1
    assert new_worker == old_worker.replace(original_line, replacement)
    observation = json.loads((BASE / 'repair_run/observation.json').read_text())
    assert observation['remote_status'] == 'ERROR' and observation['source_matches_accepted']
    assert observation['failure_records']['mars_resource_pilot/failure.json']['error_type'] == 'RuntimeError'
    logs = BASE / 'repair_run/output' / observation['attempt_directory'] / 'mars_resource_worker.log'
    text = logs.read_text()
    assert 'torch.cuda.reset_peak_memory_stats(device)' in text and 'RuntimeError: Invalid device argument' in text
    modules['mars_resource_pilot.py'] = new_worker
    contract = json.loads(modules['pilot_contract.json'])
    contract.update(schema='mars_resource_pilot_execution_v4_cuda_initialized',
                    repair='Initialize CUDA before resetting allocator peak counters')
    contract['module_sha256'] = {name: hashlib.sha256(s.encode()).hexdigest()
                                 for name, s in modules.items() if name != 'pilot_contract.json'}
    modules['pilot_contract.json'] = json.dumps(contract, indent=2) + '\n'
    code = 'MODULES=' + repr(modules) + '\n' + (
        "exec(compile(MODULES['mars_pilot_driver.py'], 'mars_pilot_driver.py', 'exec'))\n")
    ast.parse(code)
    code_cell.update(source=code, outputs=[], execution_count=None)
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE / 'pilot.ipynb').write_text(json.dumps(old, indent=2) + '\n')
    meta = json.loads((old_path.parent / 'kernel-metadata.json').read_text())
    meta['id_no'] = 138052662
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    record = {'status': 'DIAGNOSED_CUDA_INITIALIZATION_FIX_PACKAGED',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'code_cell_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'notebook_sha256': sha(PACKAGE / 'pilot.ipynb'),
              'metadata_sha256': sha(PACKAGE / 'kernel-metadata.json'),
              'original_worker_log_sha256': sha(logs),
              'previous_observation_sha256': sha(BASE / 'repair_run/observation.json'),
              'model_preprocessing_cohort_and_thresholds_unchanged': all(
                  previous[n] == modules[n] for n in previous
                  if n not in {'mars_resource_pilot.py', 'pilot_contract.json'}),
              'worker_change_is_only_cuda_initialization_and_logging': True,
              'torch_source_references': [
                  'https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/cuda/memory.py',
                  'https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/c10/cuda/CUDACachingAllocator.cpp'],
              'cuda_fix_execution_verified': False, 'gpu_launched': False}
    (BASE / 'cuda_fix_build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
