"""Preserve accepted V2 and package a restart/logging repair for review."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
PACKAGE = ROOT / 'public_kernels/mars_resource_pilot_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    launch = json.loads((RUN / 'resource_launch.json').read_text())
    old = ROOT / 'public_kernels/mars_resource_pilot_v2/pilot.py'
    assert sha(old) == launch['code_sha256'], 'Preserve the accepted source'
    modules = ast.literal_eval(ast.parse(old.read_text()).body[0].value)
    original_contract = json.loads(modules['pilot_contract.json'])
    old_target = "TARGET='/kaggle/working/mars_pilot_dependencies'"
    assert modules['check_decoders.py'].count(old_target) == 1
    modules['check_decoders.py'] = modules['check_decoders.py'].replace(
        old_target, "import os\nTARGET=os.environ['MARS_PILOT_DEPENDENCIES']")
    modules['mars_pilot_driver.py'] = (ROOT / 'inference/mars_pilot_driver.py').read_text()
    contract = dict(original_contract)
    contract.update(schema='mars_resource_pilot_execution_v3_restart_safe',
                    supersedes_source_sha256=sha(old),
                    attempt_output_rule='exclusive_lock_and_fresh_attempt_directory')
    contract['module_sha256'] = {name: hashlib.sha256(source.encode()).hexdigest()
                                 for name, source in modules.items()
                                 if name != 'pilot_contract.json'}
    modules['pilot_contract.json'] = json.dumps(contract, indent=2) + '\n'
    code = 'MODULES=' + repr(modules) + '\n' + (
        "exec(compile(MODULES['mars_pilot_driver.py'], 'mars_pilot_driver.py', 'exec'))\n")
    ast.parse(code)
    meta = json.loads((old.parent / 'kernel-metadata.json').read_text())
    meta.update(id='adityakumarrai0401/rsna-knee-mars-resource-pilot-v3-restart-safe',
                title='RSNA Knee MARS Resource Pilot V3 Restart Safe')
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE / 'pilot.py').write_text(code)
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    record = {'status': 'RESTART_REPAIR_PACKAGED_NOT_LAUNCHED',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'accepted_v2_source_sha256': sha(old),
              'source_sha256': sha(PACKAGE / 'pilot.py'),
              'metadata_sha256': sha(PACKAGE / 'kernel-metadata.json'),
              'numerical_modules_unchanged': all(
                  contract['module_sha256'][n] == h
                  for n, h in original_contract['module_sha256'].items()
                  if n != 'check_decoders.py'),
              'cohort_preprocessing_and_encoder_unchanged': True,
              'known_fixed_bug': 'FileExistsError on preserved interactive working directories',
              'unresolved_error': 'First worker CalledProcessError requires its failure.json or traceback',
              'gpu_launched': False, 'competition_submission_enabled': False}
    (RUN / 'restart_patch_build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
