"""Fixed raw-DICOM parity and expert prediction diagnostic; no submission."""
import csv
import gc
import hashlib
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import specialist_preprocess as prep
from specialist_cache_capture import CacheCapture

# __PROBE_CONTRACT__
OUT = Path('/kaggle/working/public0033_probe')
OUT.mkdir(exist_ok=True)
START = time.monotonic()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def guard():
    if time.monotonic() - START > 3300:
        raise TimeoutError('bounded specialist diagnostic expired')
    rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmRSS:'))
    if rss > 18 * (1 << 30):
        raise RuntimeError('host resident memory exceeds18GiB')
    if any(torch.cuda.max_memory_allocated(i) > 12 * (1 << 30) for i in range(torch.cuda.device_count())):
        raise RuntimeError('GPU memory guard')


def load_runtime():
    candidates = list(Path('/kaggle/input').glob('*/public0033_runtime.py'))
    candidates += list(Path('/kaggle/input').glob('*/*/public0033_runtime.py'))
    candidates += list(Path('/kaggle/input').glob('*/*/*/public0033_runtime.py'))
    candidates = [p for p in set(candidates) if sha(p) == CONTRACT['runtime_sha256']]
    if len(candidates) != 1:
        raise RuntimeError('one exact mounted specialist runtime required')
    path = candidates[0]
    spec = importlib.util.spec_from_file_location('p33_runtime', path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    manifest = runtime.verify_bundle(path.parent)
    if manifest['checkpoint']['sha256'] != CONTRACT['checkpoint_sha256']:
        raise RuntimeError('specialist checkpoint changed')
    runtime._require_t4x2()
    return runtime, path.parent, manifest


def prepare(ids, split, tag):
    guard()
    prep.STUDY_SELECTION = set(ids)
    headers = prep.annotate(prep.walk(split + '_series'))
    series = pd.read_csv(prep.ROOT / (split + '_series.csv'), dtype=str)
    plane = dict(zip(series['SeriesInstanceUID'], series['Anatomical_Plane']))
    slots = prep.pick_slots(headers, plane)
    if set(slots) != set(ids):
        raise RuntimeError('raw-DICOM study coverage differs')
    prep.DECODE_FAILED.clear()
    result = prep.build_cache(slots, plane, prep.lat_of(headers, tag), tag)
    if prep.DECODE_FAILED:
        raise RuntimeError('decode failure; partial or filled predictions ineligible')
    capture = CacheCapture(CONTRACT['input_config'])
    actual = {'band': list(prep.SLICE_BAND), 'crop_mm': prep.CROP_MM, 'group': prep.GROUP,
              'img': prep.IMG, 'rules': prep.RULES, 'slices': prep.CACHE_SLICES,
              'slots': [s[0] for s in prep.SLOTS]}
    capture.observe(result, actual)
    record = capture.require_record()
    if set(record['studies']) != set(ids):
        raise RuntimeError('prepared identities differ')
    guard()
    return record


def predict(runtime, root, record):
    cache, mask, ids = runtime._validate_cache_inputs(record['cache'], record['mask'], record['studies'])
    partitions = runtime._balanced_contiguous_partitions(len(ids))
    prediction = np.empty((len(ids), 12), np.float32)
    receipts = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runtime._worker_predict, root=root, cache=cache, mask=mask,
                               indices=partition, device_index=device) for device, partition in enumerate(partitions)]
        for future in futures:
            indices, values, receipt = future.result()
            prediction[indices] = values
            receipts.append(receipt)
    guard()
    return prediction, receipts


def write_predictions(name, ids, prediction):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['StudyInstanceUID'] + CONTRACT['targets'])
        writer.writerows([[uid] + row.tolist() for uid, row in zip(ids, prediction)])


def main():
    torch.set_num_threads(2)
    for filename, expected in CONTRACT['official_file_sha256'].items():
        if sha(prep.ROOT / filename) != expected:
            raise RuntimeError('official input changed: ' + filename)
    cfg = CONTRACT['input_config']
    prep.IMG = prep.CACHE_IMG = cfg['img']
    prep.CROP_MM = cfg['crop_mm']
    prep.CACHE_SLICES = cfg['slices']
    prep.GROUP = cfg['group']
    prep.SLICE_BAND = tuple(cfg['band'])
    prep.RULES = dict(cfg['rules'])
    if [s[0] for s in prep.SLOTS] != cfg['slots']:
        raise RuntimeError('original recovered slot layout differs')
    prep.ORDER_CACHE = ''
    runtime, root, bundle = load_runtime()
    reference_ids = [r['StudyInstanceUID'] for r in CONTRACT['reference_predictions']]
    print(json.dumps({'stage': 'reference_start', 'studies': len(reference_ids)}), flush=True)
    reference = prepare(reference_ids, 'test', 'parity3')
    values, reference_workers = predict(runtime, root, reference)
    expected = {r['StudyInstanceUID']: r for r in CONTRACT['reference_predictions']}
    target_indices = [CONTRACT['targets'].index(t) for t in ['Medial Meniscus', 'Lateral Meniscus']]
    wanted = np.asarray([[float(expected[uid][t]) for t in ['Medial Meniscus', 'Lateral Meniscus']]
                         for uid in reference['studies']])
    error = float(np.max(np.abs(values[:, target_indices] - wanted)))
    mask_sha = hashlib.sha256(np.ascontiguousarray(reference['mask']).tobytes()).hexdigest()
    parity = {'status': 'PASS' if error <= 1e-6 and mask_sha == CONTRACT['reference_mask_sha256'] else 'FAIL',
              'max_abs_prediction_error': error, 'tolerance': 1e-6, 'mask_sha256': mask_sha,
              'workers': reference_workers}
    (OUT / 'parity.json').write_text(json.dumps(parity, indent=2) + '\n')
    write_predictions('reference_predictions.csv', reference['studies'], values)
    if parity['status'] != 'PASS':
        raise RuntimeError('original numerical reference mismatch; expert scores must remain unseen')
    print(json.dumps({'stage': 'reference_pass', **parity}), flush=True)
    del reference, values
    gc.collect()
    guard()
    expert = prepare(CONTRACT['expert_ids'], 'train', 'expert58')
    predictions, workers = predict(runtime, root, expert)
    write_predictions('expert_predictions.csv', expert['studies'], predictions)
    manifest = {'status': 'COMPLETE', 'contract': CONTRACT, 'reference_parity': parity,
                'expert_studies': len(expert['studies']), 'workers': workers,
                'elapsed_seconds': time.monotonic() - START, 'fallback': 0,
                'expert_scores_computed_remotely': False, 'competition_submission_enabled': False,
                'public_score': None, 'bundle_manifest_sha256': sha(root / 'bundle_manifest.json'),
                'artifacts': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                              for p in OUT.iterdir() if p.is_file()}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', 'elapsed_seconds': manifest['elapsed_seconds'], 'expert_scores': None}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (OUT / 'failure.json').write_text(json.dumps({'status': 'INCOMPLETE_INELIGIBLE',
            'error_type': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.monotonic() - START}) + '\n')
        raise
