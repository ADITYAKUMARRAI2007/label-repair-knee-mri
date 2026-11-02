"""Screen the now-complete E04 control with the already frozen10% blend."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_e05_complete_oof import ROOT, audited_predictions, paired_interval, sha


def main():
    run = ROOT / 'runs/e05_confidence_ablation'
    protocol_path = run / 'complete_blend_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['candidate_weight'] == .1 and protocol['baseline_weight'] == .9
    assert 'original_control' in protocol['arms']
    audit_path = run / 'compute_review_control_audit.json'
    frame, audit = audited_predictions(audit_path, 'e04_independent_fold_training_audit_v1')
    ids, targets = protocol['cohort_ids'], protocol['target_order']
    assert len(ids) == 58 and set(frame.StudyInstanceUID) == set(ids)
    baseline_path = ROOT / protocol['baseline_path']
    assert sha(baseline_path) == protocol['baseline_sha256']
    baseline = pd.read_csv(baseline_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, targets]
    control = frame.set_index('StudyInstanceUID').loc[ids, targets]
    y = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, targets].to_numpy()
    assert np.isfinite(control.to_numpy()).all() and ((control >= 0) & (control <= 1)).all().all()
    baseline_rank = baseline.rank(method='average', pct=True).to_numpy()
    mixture = .9 * baseline_rank + .1 * control.rank(method='average', pct=True).to_numpy()
    base_auc, mixed_auc = roc_auc_score(y, baseline_rank, average=None), roc_auc_score(y, mixture, average=None)
    assert abs(roc_auc_score(y, control.to_numpy()) - audit['expert_macro_auc']) < 1e-12
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    rows = {r['StudyInstanceUID']: r for r in frozen['rows']}
    groups = np.array([rows[u]['report_group'] for u in ids])
    interval, draws = paired_interval(y, baseline_rank, mixture, groups)
    report = {'status': 'COMPLETE_E04_FIXED_BLEND_DIAGNOSTIC',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'expert_studies': 58, 'folds': [0, 1, 2, 3, 4],
              'owned_control_auc': audit['expert_macro_auc'],
              'baseline_auc': float(base_auc.mean()), 'fixed10percent_blend_auc': float(mixed_auc.mean()),
              'delta': float((mixed_auc - base_auc).mean()),
              'paired_group_95_interval': interval, 'valid_bootstrap_draws': draws,
              'per_target_delta': dict(zip(targets, (mixed_auc-base_auc).tolist())),
              'audit_sha256': sha(audit_path), 'protocol_sha256': sha(protocol_path),
              'source_sha256': sha(ROOT / 'scripts/screen_e04_complete_blend.py'),
              'weight_search_performed': False, 'competition_submission_enabled': False,
              'checkpoint_bytes_independently_verified': audit['checkpoint_bytes_independently_verified_locally'],
              'complete_E05_candidate_compared': False, 'public_score_prediction': None,
              'limitations': protocol['limitations'],
              'decision': 'REVIEW_ONLY_NO_AUTOMATIC_PROMOTION' if mixed_auc.mean()>base_auc.mean()
                          else 'REJECT_THIS_FIXED_CONTROL_BLEND'}
    (run / 'complete_control_blend_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_target_delta','limitations']}))


if __name__ == '__main__':
    main()
