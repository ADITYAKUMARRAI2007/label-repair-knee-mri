"""Evaluate complete held-out MARS heads and the one prespecified rank blend."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_e05_complete_oof import ROOT, sha, paired_interval

BASE = ROOT / 'research/foundation_refresh_20261010/MARS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = BASE / 'classifier_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['fixed_rank_blend'] == {'baseline': .9, 'candidate': .1,
                                           'target_weights_searched': False}
    for name, expected in protocol['sources_sha256'].items():
        assert sha(ROOT / name) == expected, name
    receipt_path = args.training_output / 'training_receipt.json'
    if not receipt_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_ACTUAL_COMPLETE_FIVE_FOLD_TRAINING',
                          'accuracy_measured': False, 'competition_submission_enabled': False}))
        return
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'FIVE_FROZEN_MARS_HEADS_AND_COMPLETE_OOF_TRAINED'
    assert receipt['protocol_sha256'] == sha(protocol_path) and receipt['studies'] == 4407
    assert receipt['competition_submission_enabled'] is False
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    rows = frozen['rows']; ids = [r['StudyInstanceUID'] for r in rows]
    by_id = {r['StudyInstanceUID']: r for r in rows}
    fold_receipts = json.loads((args.training_output / 'fold_receipts.json').read_text())
    assert [r['fold'] for r in fold_receipts] == list(range(5))
    for item in fold_receipts:
        f = item['fold']; validation = [r['StudyInstanceUID'] for r in rows if r['fold'] == f]
        groups = {by_id[uid]['report_group'] for uid in validation}
        training = [r['StudyInstanceUID'] for r in rows if r['report_group'] not in groups]
        assert item['validation_ids'] == validation and item['training_ids'] == training
        assert item['completed_epoch'] == 30
        assert item['seed'] == protocol['seed'] + 1000 * (f + 1)
        assert sha(args.training_output / f'mars_head_fold{f}.pt') == item['checkpoint_sha256']
    history = json.loads((args.training_output / 'history.json').read_text())
    assert [(r['fold'], r['epoch']) for r in history] == [
        (f, e) for f in range(5) for e in range(1, 31)]
    assert all(np.isfinite(r['loss']) and r['seconds'] > 0 for r in history)
    prediction_path = args.training_output / 'mars_oof_predictions.csv'
    assert sha(prediction_path) == receipt['oof_predictions_sha256']
    frame = pd.read_csv(prediction_path, dtype={'StudyInstanceUID': str})
    targets = protocol['target_order']
    assert frame.columns.tolist() == ['StudyInstanceUID', *targets]
    assert frame.StudyInstanceUID.tolist() == ids and frame.StudyInstanceUID.is_unique
    all_pred = frame[targets].to_numpy()
    assert np.isfinite(all_pred).all() and ((all_pred >= 0) & (all_pred <= 1)).all()
    experts = sorted(r['StudyInstanceUID'] for r in rows if r['expert'])
    assert len(experts) == 58
    candidate = frame.set_index('StudyInstanceUID').loc[experts, targets].to_numpy()
    baseline_path = ROOT / protocol['baseline_path']
    assert sha(baseline_path) == protocol['baseline_sha256']
    baseline = pd.read_csv(baseline_path, dtype={'StudyInstanceUID': str}).set_index(
        'StudyInstanceUID').loc[experts, targets].to_numpy()
    truth = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index(
        'StudyInstanceUID').loc[experts, targets].to_numpy()
    assert np.isin(truth, [0, 1]).all()
    b_rank = pd.DataFrame(baseline).rank(method='average', pct=True).to_numpy()
    c_rank = pd.DataFrame(candidate).rank(method='average', pct=True).to_numpy()
    blend = .9 * b_rank + .1 * c_rank
    base_auc = roc_auc_score(truth, b_rank, average=None)
    candidate_auc = roc_auc_score(truth, candidate, average=None)
    blend_auc = roc_auc_score(truth, blend, average=None)
    groups = np.array([by_id[u]['report_group'] for u in experts])
    interval, draws = paired_interval(truth, b_rank, blend, groups)
    delta = float(blend_auc.mean() - base_auc.mean())
    qualifies = delta > .001 and interval[0] > 0
    result = {'status': 'COMPLETE_MARS_OOF_AND_FIXED_BLEND_DIAGNOSTIC',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'expert_studies': 58, 'complete_five_fold_oof': True,
              'baseline_macro_auc': float(base_auc.mean()),
              'candidate_macro_auc': float(candidate_auc.mean()),
              'fixed_blend_macro_auc': float(blend_auc.mean()), 'fixed_blend_delta': delta,
              'paired_report_group_95_interval': interval, 'valid_bootstrap_draws': draws,
              'per_target': {t: {'baseline': float(base_auc[i]), 'candidate': float(candidate_auc[i]),
                                 'fixed_blend': float(blend_auc[i]),
                                 'blend_delta': float(blend_auc[i] - base_auc[i])}
                             for i, t in enumerate(targets)},
              'weight_search_performed': False,
              'decision': 'REVIEW_SERVING_AND_EXECUTION_PARITY_BEFORE_SUBMISSION' if qualifies
                          else 'REJECT_THIS_FIXED_MARS_BLEND_NO_SUBMISSION',
              'protocol_sha256': sha(protocol_path), 'training_receipt_sha256': sha(receipt_path),
              'oof_predictions_sha256': sha(prediction_path), 'public_score_prediction': None,
              'competition_submission_enabled': False,
              'execution_provenance_requires_separate_review': True,
              'limitations': protocol['limitations']}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['per_target', 'limitations']}))


if __name__ == '__main__':
    main()
