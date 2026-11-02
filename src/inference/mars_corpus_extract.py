"""Extract the complete frozen feature corpus only after a measured cost review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    assert review['decision'] == 'APPROVE_MEASURED_MARS_FULL_FEATURE_EXTRACTION'
    assert review['numerical_pilot_passed'] and review['complete_worker_cost_reviewed']
    assert review['required_studies'] == 4407 and review['shard_count'] == 2
    assert review['automatic_classifier_training'] is False
    assert args.shard in range(2)
    import numpy as np
    import torch
    from mars_feature_cache import MarsFeatureCache
    from mars_study import build_study
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f'cuda:{args.shard}')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    torch.cuda.init(); torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    args.out.mkdir(exist_ok=True)
    started = time.monotonic()
    budget = int(review['maximum_worker_seconds'])
    assert 0 < budget < review['platform_timeout_seconds']
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Reviewed extraction budget exhausted')))
    signal.alarm(budget)
    current = None
    try:
        series = args.input / 'train_series.csv'
        assert sha(series) == review['train_series_sha256']
        by_uid = {}
        with series.open() as handle:
            for row in csv.DictReader(handle):
                by_uid.setdefault(row['StudyInstanceUID'], []).append(row)
        uids = sorted(by_uid)
        assert len(uids) == len(set(uids)) == 4407
        assert hashlib.sha256(json.dumps(uids).encode()).hexdigest() == review['ordered_study_ids_sha256']
        selected = uids[args.shard::2]
        root = next(p for p in [args.input / 'train_series', args.input / 'train_images'] if p.is_dir())
        checkpoint = args.assets / 'MARS.pt'
        assert sha(checkpoint) == review['checkpoint_sha256']
        cache = MarsFeatureCache(args.out / 'features', checkpoint, str(device))
        assert cache.contract_sha == review['extraction_contract_sha256'], 'Pilot/corpus extraction contracts differ'
        records = []
        for index, uid in enumerate(selected):
            current = uid; before = time.monotonic()
            volumes, present, input_receipt = build_study(uid, by_uid[uid], root)
            feature, mask, record, reused = cache.extract(uid, volumes, present, input_receipt)
            torch.cuda.synchronize(device)
            assert feature.shape == (3, 768, 3, 3, 3) and np.isfinite(feature).all()
            records.append({'study_id': uid, 'cache_sha256': sha(cache.path(uid)),
                            'feature_sha256': record['feature_sha256'],
                            'input_sha256': record['input_sha256'],
                            'seconds': time.monotonic() - before, 'cache_reused': reused})
            if (index + 1) % 20 == 0:
                print(json.dumps({'stage': 'full_frozen_features', 'shard': args.shard,
                                  'completed': index + 1, 'required': len(selected)}), flush=True)
        assert [r['study_id'] for r in records] == selected
        receipt = {'status': 'COMPLETE_MARS_FEATURE_SHARD', 'shard': args.shard,
                   'shard_count': 2, 'required_studies': len(selected),
                   'review_sha256': sha(args.review), 'extraction_contract_sha256': cache.contract_sha,
                   'records': records, 'elapsed_seconds': time.monotonic() - started,
                   'gpu_peak_bytes': torch.cuda.max_memory_allocated(device),
                   'label_values_used': False, 'classifier_training_started': False}
        (args.out / f'shard{args.shard}_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except BaseException as error:
        (args.out / f'shard{args.shard}_failure.json').write_text(json.dumps({
            'status': 'CORPUS_EXTRACTION_INCOMPLETE', 'shard': args.shard, 'study_id': current,
            'error_type': type(error).__name__, 'error_message': str(error),
            'elapsed_seconds': time.monotonic() - started}) + '\n')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
