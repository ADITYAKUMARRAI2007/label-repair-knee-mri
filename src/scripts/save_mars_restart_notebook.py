"""Quick-save valid notebook JSON after the script conversion failure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
PACKAGE = ROOT / 'public_kernels/mars_resource_restart_notebook'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    built = json.loads((RUN / 'restart_patch_build.json').read_text())
    source_path = ROOT / 'public_kernels/mars_resource_pilot_v3/pilot.py'
    assert sha(source_path) == built['source_sha256'] and built['numerical_modules_unchanged']
    assert json.loads((RUN / 'restart_patch_check.json').read_text())['status'].startswith('PASS_')
    source = source_path.read_text()
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.13'}},
                'cells': [{'cell_type': 'markdown', 'id': 'purpose', 'metadata': {},
                           'source': 'MARS resource pilot restart repair. Preserves earlier outputs and shows worker tracebacks. No classifier accuracy or submission is produced.'},
                          {'cell_type': 'code', 'id': 'pilot-driver', 'metadata': {},
                           'execution_count': None, 'outputs': [], 'source': source}]}
    PACKAGE.mkdir(exist_ok=True)
    path = PACKAGE / 'pilot.ipynb'; path.write_text(json.dumps(notebook, indent=2) + '\n')
    meta = json.loads((source_path.parent / 'kernel-metadata.json').read_text())
    meta.update(id='adityakumarrai0401/rsna-knee-mars-restart-repair',
                title='RSNA Knee MARS Restart Repair', code_file='pilot.ipynb', kernel_type='notebook')
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    record = {'status': 'RESTART_REPAIR_NOTEBOOK_PREPARED',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'kernel': meta['id'], 'notebook_sha256': sha(path),
              'code_cell_sha256': built['source_sha256'],
              'requested_execution_type': 'QUICK_SAVE', 'gpu_run_requested': False,
              'original_worker_failure_resolved': False, 'competition_submission_enabled': False}
    if not args.save:
        print(json.dumps(record)); return
    receipt = RUN / 'restart_notebook_save.json'
    with receipt.open('x') as handle:
        json.dump(record, handle, indent=2); handle.write('\n')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(55)
    try:
        api = KaggleApi(); api.authenticate()
        request = ApiSaveKernelRequest()
        request.slug = meta['id']; request.new_title = meta['title']; request.text = path.read_text()
        request.language = 'python'; request.kernel_type = 'notebook'; request.is_private = True
        request.enable_gpu = True; request.enable_tpu = False; request.enable_internet = False
        request.dataset_data_sources = meta['dataset_sources']
        request.competition_data_sources = meta['competition_sources']
        request.kernel_data_sources = []; request.model_data_sources = []
        request.docker_image = meta['docker_image']; request.machine_shape = meta['machine_shape']
        request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record['save_response'] = response.to_dict()
        assert not response.error
        # Quick Save may return empty identifiers. A separate source pull proves persistence.
        record['status'] = 'QUICK_SAVE_RETURNED_VERIFY_REMOTE_SOURCE'
    except BaseException as error:
        record.update(status='SAVE_OUTCOME_REQUIRES_REVIEW', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0); receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
