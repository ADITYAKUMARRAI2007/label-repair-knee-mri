"""Fit independently initialized heads on verified complete frozen MARS features."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from mars_spatial_head import MarsSpatialHead


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def split_rows(rows, fold):
    heldout = np.array([r['fold'] == fold for r in rows])
    groups = np.array([r['report_group'] for r in rows])
    training = np.flatnonzero(~np.isin(groups, np.unique(groups[heldout])))
    validation = np.flatnonzero(heldout)
    assert len(training) and len(validation)
    assert not (set(groups[training]) & set(groups[validation]))
    return training, validation


def load_features(directory, rows, protocol):
    shape = tuple(protocol['features_shape'])
    features = np.empty((len(rows), *shape), np.float32)
    present = np.empty((len(rows), 3), np.bool_)
    extraction_contract, receipts = None, []
    for i, row in enumerate(rows):
        uid = row['StudyInstanceUID']; path = directory / (uid + '.npz')
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {'features', 'present', 'record'}, path.name
            f, mask = archive['features'], archive['present']
            record = json.loads(str(archive['record'].item()))
        c = record['extraction_contract']
        csha = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        assert record['study_id'] == uid and csha == record['contract_sha256']
        assert c['checkpoint_sha256'] == protocol['checkpoint_sha256']
        assert c['features_shape'] == list(shape) and c['planes'] == protocol['planes']
        assert c['dtype'] == 'float32' and c['autocast'] is False and not c['labels_used']
        assert not c['global_pooling_applied']
        if extraction_contract is None:
            extraction_contract = csha
        assert csha == extraction_contract, 'Mixed extraction runtimes/sources rejected'
        assert f.shape == shape and f.dtype == np.float32 and np.isfinite(f).all()
        assert mask.shape == (3,) and mask.dtype == np.bool_ and mask.any()
        assert np.all(f[~mask] == 0)
        assert digest(f) == record['feature_sha256'] and digest(mask) == record['present_sha256']
        identity = record['input_identity']
        assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == record['input_sha256']
        assert identity['input_receipt']['study_id'] == uid
        assert identity['input_receipt']['label_values_used'] is False
        assert identity['input_receipt']['fallback_predictions_used'] is False
        features[i], present[i] = f, mask
        receipts.append({'uid': uid, 'cache_sha256': sha(path),
                         'input_sha256': record['input_sha256'],
                         'feature_sha256': record['feature_sha256']})
        if (i + 1) % 256 == 0:
            print(json.dumps({'stage': 'feature_integrity', 'studies': i + 1}), flush=True)
    return features, present, extraction_contract, receipts


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--resource-approval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol['schema'] == 'mars_frozen_spatial_classifier_v1'
    assert protocol['epochs'] == 30 and protocol['folds'] == list(range(5))
    assert protocol['encoder_trainable'] is False
    approval = json.loads(args.resource_approval.read_text())
    assert approval['decision'] == 'APPROVE_COMPLETE_FROZEN_FEATURE_CLASSIFIER'
    assert approval['classifier_protocol_sha256'] == sha(args.protocol)
    assert approval['complete_study_coverage'] == 4407 and approval['numerical_and_cost_review_passed']
    for relative, expected in protocol['sources_sha256'].items():
        assert sha(args.workspace / relative) == expected, relative
    assert sha(Path(__file__)) == protocol['sources_sha256']['training/mars_classifier_train.py']
    assert sha(Path(__file__).with_name('mars_spatial_head.py')) == protocol['sources_sha256']['training/mars_spatial_head.py']
    frozen = json.loads((args.workspace / 'runs/E04_contract/e04_contract.json').read_text())
    rows = frozen['rows']; ids = [r['StudyInstanceUID'] for r in rows]
    assert len(ids) == len(set(ids)) == 4407 and ids == sorted(ids)
    y = np.load(args.workspace / 'runs/E04_contract/e04_targets.npy', allow_pickle=False)
    weights = np.load(args.workspace / 'runs/E04_contract/e04_weights.npy', allow_pickle=False)
    assert y.shape == weights.shape == (4407, 12)
    assert np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    assert np.isfinite(weights).all() and ((weights >= .25) & (weights <= 2)).all()
    features, masks, extraction_sha, input_receipts = load_features(args.features, rows, protocol)
    assert extraction_sha == approval['extraction_contract_sha256']
    args.out.mkdir(exist_ok=False)
    (args.out / 'feature_inputs.json').write_text(json.dumps(input_receipts, indent=2) + '\n')
    device = torch.device(args.device)
    assert device.type == 'cuda' and torch.cuda.is_available(), 'Actual training requires allocated CUDA'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    oof = np.full((4407, 12), np.nan, np.float32)
    history, fold_receipts = [], []
    for fold in protocol['folds']:
        seed = protocol['seed'] + 1000 * (fold + 1)
        seed_all(seed)
        train, validation = split_rows(rows, fold)
        model = MarsSpatialHead().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                      weight_decay=protocol['weight_decay'])
        for epoch in range(1, protocol['epochs'] + 1):
            seed_all(seed + epoch)
            model.train(); order = np.random.permutation(train)
            started = time.monotonic(); loss_sum, batches = 0., 0
            for start in range(0, len(order), protocol['batch_studies']):
                ix = order[start:start + protocol['batch_studies']]
                x = torch.from_numpy(features[ix]).to(device)
                mask = torch.from_numpy(masks[ix]).to(device)
                target = torch.from_numpy(y[ix]).to(device)
                confidence = torch.from_numpy(weights[ix]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, mask)
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction='none') * confidence).mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite classifier loss')
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError('Nonfinite classifier gradient')
                optimizer.step(); loss_sum += float(loss.detach()); batches += 1
            item = {'fold': fold, 'epoch': epoch, 'loss': loss_sum / batches,
                    'seconds': time.monotonic() - started}
            history.append(item); print(json.dumps(item), flush=True)
        checkpoint = args.out / f'mars_head_fold{fold}.pt'
        torch.save({'model': model.cpu().state_dict(), 'completed_epoch': 30, 'fold': fold,
                    'protocol_sha256': sha(args.protocol), 'extraction_contract_sha256': extraction_sha},
                   checkpoint)
        model.to(device).eval()
        with torch.inference_mode():
            for start in range(0, len(validation), protocol['batch_studies']):
                ix = validation[start:start + protocol['batch_studies']]
                pred = model(torch.from_numpy(features[ix]).to(device),
                             torch.from_numpy(masks[ix]).to(device)).sigmoid().cpu().numpy()
                assert np.isfinite(pred).all()
                oof[ix] = pred
        fold_receipts.append({'fold': fold, 'seed': seed, 'training_ids': [ids[i] for i in train],
                              'validation_ids': [ids[i] for i in validation],
                              'checkpoint_sha256': sha(checkpoint), 'completed_epoch': 30})
        del optimizer, model
    assert np.isfinite(oof).all()
    prediction_path = args.out / 'mars_oof_predictions.csv'
    with prediction_path.open('w') as handle:
        writer = csv.writer(handle); writer.writerow(['StudyInstanceUID', *protocol['target_order']])
        writer.writerows([uid, *pred.tolist()] for uid, pred in zip(ids, oof))
    (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.out / 'fold_receipts.json').write_text(json.dumps(fold_receipts, indent=2) + '\n')
    (args.out / 'training_receipt.json').write_text(json.dumps({
        'status': 'FIVE_FROZEN_MARS_HEADS_AND_COMPLETE_OOF_TRAINED',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(args.protocol), 'resource_approval_sha256': sha(args.resource_approval),
        'extraction_contract_sha256': extraction_sha, 'studies': len(ids),
        'oof_predictions_sha256': sha(prediction_path),
        'competition_submission_enabled': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
