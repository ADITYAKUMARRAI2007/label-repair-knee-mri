"""Save a private repaired script with explicit QUICK_SAVE; never request a run."""
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
PACKAGE = ROOT / 'public_kernels/mars_resource_pilot_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    if args.save:
        raise RuntimeError('Raw-script Quick Save fails notebook conversion; use save_mars_restart_notebook.py')
    built = json.loads((RUN / 'restart_patch_build.json').read_text())
    check = json.loads((RUN / 'restart_patch_check.json').read_text())
    assert built['numerical_modules_unchanged'] and not built['gpu_launched']
    assert check['status'] == 'PASS_RESTART_AND_WORKER_DIAGNOSTIC_REPRODUCTION'
    meta = json.loads((PACKAGE / 'kernel-metadata.json').read_text())
    assert sha(PACKAGE / meta['code_file']) == built['source_sha256']
    assert sha(PACKAGE / 'kernel-metadata.json') == built['metadata_sha256']
    assert meta['is_private'] and not meta['enable_internet']
    record = {'status': 'PRIVATE_QUICK_SAVE_REVIEW_READY',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'kernel': meta['id'], 'source_sha256': built['source_sha256'],
              'requested_execution_type': 'QUICK_SAVE',
              'gpu_run_requested': False, 'competition_submission_enabled': False,
              'original_worker_failure_resolved': False}
    if not args.save:
        print(json.dumps(record)); return
    receipt = RUN / 'restart_patch_save.json'
    with receipt.open('x') as handle:
        json.dump(record, handle, indent=2); handle.write('\n')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(55)
    try:
        api = KaggleApi(); api.authenticate()
        request = ApiSaveKernelRequest()
        request.slug = meta['id']; request.new_title = meta['title']
        request.text = (PACKAGE / meta['code_file']).read_text()
        request.language = meta['language']; request.kernel_type = meta['kernel_type']
        request.is_private = True; request.enable_gpu = meta['enable_gpu']
        request.enable_tpu = False; request.enable_internet = False
        request.dataset_data_sources = meta['dataset_sources']
        request.competition_data_sources = meta['competition_sources']
        request.kernel_data_sources = meta['kernel_sources']
        request.model_data_sources = meta['model_sources']
        request.category_ids = meta['keywords']; request.docker_image = meta['docker_image']
        request.machine_shape = meta['machine_shape']
        request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record['save_response'] = response.to_dict()
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        assert not response.error, 'Quick save returned an error; inspect the receipt'
        assert response.ref == meta['id'], 'Returned a different notebook ref'
        record.update(status='PRIVATE_QUICK_SAVE_ACCEPTED', version=response.version_number,
                      url=response.url, kernel_id=response.kernel_id)
    except BaseException as error:
        record.update(status='QUICK_SAVE_OUTCOME_REQUIRES_REVIEW',
                      error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
