"""Run the fixed five-head reader on native unseen test studies only."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from mars_encoder import encode_study_sequences, load_encoder
from mars_feature_cache import MarsFeatureCache
from mars_head_inference import MarsHeadInference
from mars_study import build_study


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root = Path(root)
    runtime = json.loads((root / 'test_runtime.json').read_text())
    data, assets, trained, out = [Path(runtime[k]) for k in ['data', 'assets', 'trained', 'out']]
    contract = json.loads((root / 'test_contract.json').read_text())
    protocol_path = root / 'classifier_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    serving = json.loads((root / 'serving_protocol.json').read_text())
    assert sha(protocol_path) == serving['classifier_protocol_sha256']
    assert sha(root / 'mars_head_inference.py') == serving['head_source_sha256']
    assert serving['head_aggregation'] == 'arithmetic mean of five sigmoid probabilities on unseen test studies only'
    for name, expected in contract['module_sha256'].items():
        assert sha(root / name) == expected, name
    import monai
    assert str(torch.__version__) == contract['torch_version']
    assert monai.__version__ == contract['monai_version']
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init(); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    checkpoint = assets / 'MARS.pt'
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    assert sha(trained / 'training_receipt.json') == contract['training_receipt_sha256']
    with (data / 'test.csv').open() as handle:
        ids = [row['StudyInstanceUID'] for row in csv.DictReader(handle)]
    assert len(ids) == len(set(ids)) > 0
    records = {uid: [] for uid in ids}
    with (data / 'test_series.csv').open() as handle:
        for row in csv.DictReader(handle):
            assert row['StudyInstanceUID'] in records
            records[row['StudyInstanceUID']].append(row)
    assert all(records.values())
    series_root = next(p for p in [data / 'test_series', data / 'test_images'] if p.is_dir())
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    results, reference_parity = [], None
    with tempfile.TemporaryDirectory(prefix='mars_test_features_', dir='/tmp') as temporary:
        cache = MarsFeatureCache(Path(temporary), checkpoint, 'cuda:0')
        assert cache.contract_sha == contract['extraction_contract_sha256']
        heads = MarsHeadInference(trained, protocol_path, cache.contract_sha, 'cuda:0')
        assert not (set(ids) & heads.training_ids), 'Test cohort overlaps classifier training IDs'
        for uid in ids:
            before = time.monotonic()
            volumes, present, input_receipt = build_study(uid, records[uid], series_root)
            features, mask, record, reused = cache.extract(uid, volumes, present, input_receipt)
            assert not reused
            if reference_parity is None:
                cpu_model = load_encoder(checkpoint)
                expected = encode_study_sequences(cpu_model, torch.from_numpy(volumes[None]),
                    torch.from_numpy(present[None]))[0].numpy()
                error = float(np.max(np.abs(features - expected)))
                assert np.isfinite(expected).all() and error <= .003
                reference_parity = {'study_id': uid, 'maximum_absolute_error': error,
                    'absolute_tolerance': .003, 'cpu_reference_created_inside_test_mount': True}
                del cpu_model, expected
            prediction = heads.predict_test(torch.from_numpy(features[None]),
                torch.from_numpy(mask[None]), [uid])[0]
            results.append([uid, *prediction.tolist()])
            item = {'study_id': uid, 'seconds': time.monotonic() - before,
                'present': present.tolist(), 'input_sha256': record['input_sha256'],
                'feature_sha256': record['feature_sha256']}
            with (out / 'study_receipts.jsonl').open('a') as handle:
                handle.write(json.dumps(item) + '\n')
            if len(results) % 20 == 0 or len(results) == len(ids):
                print(json.dumps({'stage': 'mars_test_inference', 'completed': len(results),
                                  'required': len(ids)}), flush=True)
    csv_path = out / 'candidate.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['StudyInstanceUID', *heads.target_order]); writer.writerows(results)
    receipt = {'status': 'COMPLETE_NATIVE_UNSEEN_TEST_FIVE_HEAD_MARS_READER',
        'studies': len(ids), 'ordered_test_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'training_receipt_sha256': sha(trained / 'training_receipt.json'),
        'protocol_sha256': sha(protocol_path), 'extraction_contract_sha256': cache.contract_sha,
        'candidate_sha256': sha(csv_path), 'cpu_cuda_encoder_parity': reference_parity,
        'gpu_peak_bytes': torch.cuda.max_memory_allocated(0),
        'elapsed_seconds': time.monotonic() - started, 'clinical_accuracy_measured': False,
        'head_aggregation': 'mean_of_five_sigmoid_probabilities', 'labels_used': False,
        'fallback_predictions_used': False, 'competition_submission_enabled': False}
    (out / 'inference_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    run(Path(__file__).resolve().parent)
