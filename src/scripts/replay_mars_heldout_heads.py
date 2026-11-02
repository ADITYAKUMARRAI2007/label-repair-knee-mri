"""Replay every OOF row through its reloaded held-out head on native features."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import mars_head_inference
from mars_head_inference import MarsHeadInference
from mars_classifier_train import load_features, split_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.training_output / 'training_receipt.json'
    if not receipt_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_ACTUAL_TRAINED_HEADS', 'parity_verified': False}))
        return
    protocol_path = args.workspace / 'classifier_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    serving_path = args.workspace / 'serving_protocol.json'
    serving = json.loads(serving_path.read_text())
    assert serving['classifier_protocol_sha256'] == sha(protocol_path)
    assert serving['head_source_sha256'] == sha(mars_head_inference.__file__)
    assert serving['dtype'] == 'float32' and not serving['tf32_allowed']
    for relative, expected in protocol['sources_sha256'].items():
        assert sha(args.workspace / relative) == expected, relative
    receipt = json.loads(receipt_path.read_text())
    rows = json.loads((args.workspace / 'runs/E04_contract/e04_contract.json').read_text())['rows']
    ids = [row['StudyInstanceUID'] for row in rows]
    assert len(ids) == len(set(ids)) == 4407 and ids == sorted(ids)
    features, present, extraction_sha, _ = load_features(args.features, rows, protocol)
    assert receipt['extraction_contract_sha256'] == extraction_sha
    prediction_path = args.training_output / 'mars_oof_predictions.csv'
    assert sha(prediction_path) == receipt['oof_predictions_sha256']
    import csv
    with prediction_path.open() as handle:
        reader = csv.reader(handle)
        assert next(reader) == ['StudyInstanceUID', *protocol['target_order']]
        saved = list(reader)
    assert [row[0] for row in saved] == ids
    expected = np.array([row[1:] for row in saved], dtype=np.float32)
    assert expected.shape == (4407, 12) and np.isfinite(expected).all()
    assert ((expected >= 0) & (expected <= 1)).all()
    assert torch.cuda.is_available(), 'Replay uses the actual CUDA serving runtime'
    torch.set_num_threads(2)
    heads = MarsHeadInference(args.training_output, protocol_path, extraction_sha, 'cuda:0')
    fold_receipts = json.loads((args.training_output / 'fold_receipts.json').read_text())
    replay = np.full_like(expected, np.nan)
    per_fold = []
    for fold in protocol['folds']:
        train, validation = split_rows(rows, fold)
        assert fold_receipts[fold]['training_ids'] == [ids[i] for i in train]
        assert fold_receipts[fold]['validation_ids'] == [ids[i] for i in validation]
        for start in range(0, len(validation), protocol['batch_studies']):
            ix = validation[start:start + protocol['batch_studies']]
            replay[ix] = heads.predict_heldout(torch.from_numpy(features[ix]),
                torch.from_numpy(present[ix]), [ids[i] for i in ix], fold)
        difference = float(np.max(np.abs(replay[validation] - expected[validation])))
        assert difference <= 1e-5, f'Fold {fold} saved/reloaded prediction mismatch: {difference}'
        per_fold.append({'fold': fold, 'studies': len(validation), 'maximum_absolute_error': difference})
        print(json.dumps({'stage': 'heldout_head_replay', **per_fold[-1]}), flush=True)
    assert np.isfinite(replay).all()
    result = {'status': 'PASS_ALL_4407_RELOADED_HELDOUT_HEAD_PREDICTIONS',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'studies': 4407,
        'maximum_absolute_error': float(np.max(np.abs(replay - expected))),
        'absolute_tolerance': 1e-5, 'folds': per_fold,
        'training_receipt_sha256': sha(receipt_path), 'oof_predictions_sha256': sha(prediction_path),
        'serving_protocol_sha256': sha(serving_path), 'head_source_sha256': sha(mars_head_inference.__file__),
        'verifier_source_sha256': sha(__file__), 'extraction_contract_sha256': extraction_sha,
        'test_ensemble_or_end_to_end_dicom_serving_verified': False,
        'weight_search_performed': False, 'competition_submission_enabled': False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
