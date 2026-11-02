"""Resume downloading one completed pilot; never launch or change GPU code."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
import requests

from audit_mars_cuda_fix import ROOT, RUN, audit, sha
from observe_mars_cuda_fix import PATTERN
from observe_coherent_anchor_pair import remote_status_name


def collect(api, kernel, output, pattern, log_path):
    """List pages individually, then download selected artifacts atomically."""
    owner, slug = kernel.split('/')
    selected, seen, token = {}, set(), None
    with api.build_kaggle_client() as client:
        for _ in range(100):
            signal.alarm(45)
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 100, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if re.search(pattern, item.file_name):
                    name = Path(item.file_name)
                    assert not name.is_absolute() and '..' not in name.parts
                    assert item.file_name not in selected
                    selected[item.file_name] = item.url
            if response.log:
                log_path.write_text(response.log)
            token = response.next_page_token
            if not token: break
            assert token not in seen; seen.add(token)
        else:
            raise AssertionError('Artifact listing did not terminate')
    signal.alarm(0)
    assert selected
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def fetch(item):
        name, url = item
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        error_type = None
        for _ in range(2):
            partial = path.with_name(path.name + '.download')
            try:
                if time.monotonic() - started > 180:
                    raise TimeoutError('Bounded artifact collection expired')
                with requests.get(url, stream=True, timeout=(5, 15)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as handle:
                        for chunk in response.iter_content(1 << 20):
                            handle.write(chunk)
                partial.replace(path)
                return name, None
            except Exception as error:
                error_type = type(error).__name__
                partial.unlink(missing_ok=True)
        return name, error_type

    failures, completed = {}, 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, item) for item in selected.items()]
        for future in as_completed(futures):
            name, error = future.result()
            if error: failures[name] = error
            else: completed += 1
            if (completed + len(failures)) % 10 == 0:
                print(json.dumps({'collected': completed, 'total': len(selected),
                                  'download_failures': len(failures)}), flush=True)
    return completed, failures


def main():
    launch = json.loads((RUN / 'launch.json').read_text())
    review = json.loads((RUN / 'reproduction_review.json').read_text())
    assert sha(RUN / 'reproduction_review.json') == launch['review_sha256']
    for relative, expected in review['source_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    api = KaggleApi(); api.authenticate()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(45)
    assert remote_status_name(api.kernels_status(launch['kernel'])) == 'COMPLETE'
    source = RUN / 'remote_source'
    api.kernels_pull(launch['kernel'], str(source), metadata=True, quiet=True)
    meta = json.loads((source / 'kernel-metadata.json').read_text())
    book = json.loads((source / meta['code_file']).read_text())
    cells = [cell for cell in book['cells'] if cell['cell_type'] == 'code']
    assert len(cells) == 1
    code = cells[0]['source']; code = ''.join(code) if isinstance(code, list) else code
    assert hashlib.sha256(code.encode()).hexdigest() == launch['code_cell_sha256']
    output = RUN / 'output'
    completed, failures = collect(api, launch['kernel'], output, PATTERN, RUN / 'remote_execution.log')
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'kernel': launch['kernel'], 'remote_status': 'COMPLETE',
              'source_matches_accepted': True, 'collected_files': completed,
              'download_failures': failures, 'gpu_run_requested': False,
              'competition_submission_enabled': False}
    if not failures:
        marker = json.loads((output / 'mars_latest_attempt.json').read_text())
        name = marker['attempt_directory']
        assert Path(name).name == name and name.startswith('mars_resource_attempt_')
        reviewed = audit(output / name)
        (RUN / 'execution_audit.json').write_text(json.dumps(reviewed, indent=2) + '\n')
        record.update(attempt_directory=name, execution_audit_status=reviewed['status'])
    record['next_action'] = 'REVIEW_MEASURED_RESOURCE_RESULT_BEFORE_FULL_CORPUS_EXTRACTION'
    (RUN / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
