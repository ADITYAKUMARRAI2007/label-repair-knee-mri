"""Bounded real-DICOM MARS resource measurement; no fitting or submission."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol['schema'] == 'mars_rsna_resource_pilot_v2_model_only'
    assert not protocol['classification_fitting_enabled']
    assert not protocol['label_values_used'] and not protocol['competition_submission_enabled']
    studies = protocol['studies']
    assert len(studies) == len(set(studies)) == 24
    args.out.mkdir(exist_ok=False)
    started = time.monotonic()
    stage, current_uid = 'startup', None
    def expire(*_):
        raise TimeoutError('Fixed resource pilot exceeded its execution budget')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(protocol['complete_worker_seconds_cap'])
    try:
        import monai
        import numpy as np
        import torch
        from mars_feature_cache import MarsFeatureCache
        from mars_study import build_study
        from mars_encoder import load_encoder, encode_study_sequences
        assert monai.__version__ == protocol['monai_version']
        if not args.device.startswith('cuda:') or not torch.cuda.is_available():
            raise RuntimeError('Resource pilot requires its explicitly allocated CUDA device')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        stage = 'cuda_initialization'
        # CUDA availability/count queries do not initialize the allocator.
        # reset_peak_memory_stats enters the allocator directly on this runtime.
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({'stage': 'cuda_memory_counter_ready',
                          'device': str(device), 'cuda_initialized': torch.cuda.is_initialized()}),
              flush=True)
        checkpoint = args.assets / 'MARS.pt'
        assert sha(checkpoint) == protocol['checkpoint_sha256']
        stage = 'checkpoint'
        cache = MarsFeatureCache(args.out / 'features', checkpoint, args.device)
        stage = 'metadata'
        series_csv = args.input / 'train_series.csv'
        assert sha(series_csv) == protocol['sources_sha256']['data/train_series.csv']
        records = {uid: [] for uid in studies}
        with series_csv.open() as handle:
            for row in csv.DictReader(handle):
                if row['StudyInstanceUID'] in records:
                    records[row['StudyInstanceUID']].append(row)
        assert all(records.values())
        series_root = next(p for p in [args.input / 'train_series', args.input / 'train_images'] if p.is_dir())
        measured = []
        parity = None
        for uid in studies:
            current_uid, stage = uid, 'full_native_dicom_decode'
            torch.cuda.synchronize(device)
            before = time.monotonic()
            volumes, present, input_receipt = build_study(uid, records[uid], series_root)
            decode_seconds = time.monotonic() - before
            reference_seconds = 0.
            if parity is None:
                # Build the reference inside the existing competition mount.
                # No MRI images or embeddings were uploaded as input assets.
                stage = 'first_study_cpu_reference'
                reference_started = time.monotonic()
                cpu_model = load_encoder(checkpoint)
                expected = encode_study_sequences(cpu_model, torch.from_numpy(volumes[None]),
                                                  torch.from_numpy(present[None]))[0].numpy()
                reference_seconds = time.monotonic() - reference_started
                del cpu_model
            stage = 'encoder_and_cache'
            encode_started = time.monotonic()
            actual, _, record, reused = cache.extract(uid, volumes, present, input_receipt)
            torch.cuda.synchronize(device)
            encoder_seconds = time.monotonic() - encode_started
            total_seconds = decode_seconds + encoder_seconds
            assert not reused
            if parity is None:
                error = float(np.max(np.abs(actual - expected)))
                parity = {'cpu_cuda_max_abs_error': error, 'finite': bool(np.isfinite(actual).all()),
                          'threshold': protocol['cuda_fixture_max_abs_error'],
                          'study_id': uid, 'cpu_reference_seconds': reference_seconds,
                          'reference_created_inside_competition_mount': True,
                          'extraction_contract': record['extraction_contract']}
                (args.out / 'fixture_parity.json').write_text(json.dumps(parity, indent=2) + '\n')
                if not parity['finite'] or error > protocol['cuda_fixture_max_abs_error']:
                    raise RuntimeError('CUDA reference feature parity failed')
                del expected
            observation = {
                'study_id': uid, 'decode_seconds': decode_seconds,
                'encoder_and_persistence_seconds': encoder_seconds,
                'excluded_one_time_cpu_reference_seconds': reference_seconds,
                'complete_study_seconds': total_seconds,
                'present_planes': int(present.sum()), 'missing_mask': (~present).tolist(),
                'feature_sha256': record['feature_sha256'], 'input_sha256': record['input_sha256'],
                'decoded_native_slices': sum(p.get('volume_receipt', {}).get('decoded_slices', 0)
                                             for p in input_receipt['planes']),
            }
            measured.append(observation)
            with (args.out / 'study_measurements.jsonl').open('a') as log:
                log.write(json.dumps(observation) + '\n')
            print(json.dumps({'stage': 'resource_pilot', 'completed': len(measured),
                              'total': len(studies), 'complete_study_seconds': total_seconds}), flush=True)
        latency = np.array([r['complete_study_seconds'] for r in measured])
        result = {
            'status': 'REAL_DICOM_CUDA_RESOURCE_PILOT_COMPLETE_NO_ACCURACY',
            'protocol_sha256': sha(args.protocol), 'completed_studies': len(measured),
            'fixture_parity': parity, 'torch_version': torch.__version__,
            'monai_version': monai.__version__, 'numpy_version': np.__version__,
            'python_version': sys.version.split()[0],
            'gpu_peak_bytes': torch.cuda.max_memory_allocated(device),
            'median_study_seconds': float(np.median(latency)),
            'mean_study_seconds': float(latency.mean()), 'p95_study_seconds': float(np.quantile(latency, .95)),
            'serial_4407_studies_projection_hours_mean_20pct_margin': float(latency.mean() * 4407 * 1.2 / 3600),
            'serial_4407_studies_projection_hours_p95_20pct_margin': float(np.quantile(latency, .95) * 4407 * 1.2 / 3600),
            'elapsed_seconds': time.monotonic() - started, 'allocated_device_count': torch.cuda.device_count(),
            'label_values_used': False, 'accuracy_measured': False,
            'classification_fitting_enabled': False, 'competition_submission_enabled': False,
            'full_corpus_continuation_enabled': False,
            'limitations': protocol['limitations'] + [
                'Projection is from a small purposive sample; it is not a guaranteed corpus runtime.',
                'Serial elapsed-hour projection is not allocated GPU-hour cost when multiple GPUs are provisioned.'],
        }
        (args.out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ['status', 'completed_studies', 'median_study_seconds']}), flush=True)
    except BaseException as error:
        (args.out / 'failure.json').write_text(json.dumps({
            'status': 'RESOURCE_PILOT_REJECTED', 'stage': stage,
            'study_id': current_uid, 'error_type': type(error).__name__,
            'elapsed_seconds': time.monotonic() - started,
            'classification_fitting_enabled': False, 'competition_submission_enabled': False,
        }, indent=2) + '\n')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
