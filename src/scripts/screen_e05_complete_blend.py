"""Fixed-weight complementarity diagnostic, only after both five-fold audits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_e05_complete_oof import ROOT, audited_predictions, paired_interval, sha
from screen_e06_complement import select_predictions, summarize

RUN = ROOT / 'runs/e05_confidence_ablation'
PROTOCOL = RUN / 'complete_blend_protocol.json'


def freeze():
    if PROTOCOL.exists():
        return json.loads(PROTOCOL.read_text())
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    baseline = ROOT / 'runs/colab_n03/output/n02_diagnostic.csv'
    protocol = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Descriptive complete five-fold complementarity screen; no weight search',
        'candidate_weight': .10, 'baseline_weight': .90,
        'weight_origin': 'Same fixed10% diagnostic used in E06; not optimized for E05',
        'baseline_path': str(baseline.relative_to(ROOT)), 'baseline_sha256': sha(baseline),
        'cohort_ids': sorted(r['StudyInstanceUID'] for r in frozen['rows'] if r['expert']),
        'target_order': frozen['target_order'],
        'required_folds': [0, 1, 2, 3, 4], 'required_experts': 58,
        'ranking': 'Within all58 expert studies per target, average ties, percentile ranks',
        'bootstrap_seed': 20261010, 'bootstrap_draws': 2000,
        'arms': ['original_control', 'uniform_candidate'],
        'known_results_at_freeze': 'Candidate folds0/1 and reused2, original controls0–3, E06 fold2 mixtures known. Folds3 candidate and4 pair unavailable.',
        'comparison_protocol_sha256': sha(RUN / 'comparison_protocol.json'),
        'reuse_amendment_sha256': sha(RUN / 'e06_control_reuse_amendment.json'),
        'competition_submission_enabled': False,
        'limitations': [
            'Gold58 has been repeatedly reused; this is not fresh independent validation.',
            'Partial model outcomes were known before this full-cohort blend protocol was written.',
            'Public-baseline training-label independence is not established.',
            'Ranking on58 studies differs from ranking on a full serving cohort.',
            'One held-out model predicts each expert; deployment would ensemble five models.',
            'Conditional report-group bootstrap excludes training and leaderboard uncertainty.',
            'A positive diagnostic does not guarantee a public or private leaderboard gain.',
        ],
    }
    with PROTOCOL.open('x') as handle:
        json.dump(protocol, handle, indent=2); handle.write('\n')
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--control-audit', type=Path)
    parser.add_argument('--candidate-audit', type=Path)
    args = parser.parse_args()
    protocol = freeze()
    assert protocol['candidate_weight'] == .10 and protocol['baseline_weight'] == .90
    assert protocol['comparison_protocol_sha256'] == sha(RUN / 'comparison_protocol.json')
    assert protocol['reuse_amendment_sha256'] == sha(RUN / 'e06_control_reuse_amendment.json')
    if args.freeze_only:
        print(json.dumps({'status': 'FROZEN_WITH_PARTIAL_RESULTS_DISCLOSED',
                          'protocol_sha256': sha(PROTOCOL), 'gpu_launched': False})); return
    assert args.control_audit and args.candidate_audit, 'Both complete five-fold audits are required'
    control, ca = audited_predictions(args.control_audit, 'e04_independent_fold_training_audit_v1')
    candidate, ea = audited_predictions(args.candidate_audit, 'e05_uniform_independent_fold_training_audit_v1')
    ids, targets = protocol['cohort_ids'], protocol['target_order']
    assert len(ids) == 58 and set(ids) == set(control.StudyInstanceUID) == set(candidate.StudyInstanceUID)
    baseline_path = ROOT / protocol['baseline_path']
    assert sha(baseline_path) == protocol['baseline_sha256']
    baseline = select_predictions(baseline_path, ids, targets)
    truth = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, targets].to_numpy()
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    rows = {r['StudyInstanceUID']: r for r in frozen['rows']}
    groups = np.array([rows[u]['report_group'] for u in ids])
    base_rank = pd.DataFrame(baseline).rank(method='average', pct=True).to_numpy()
    arms = {}
    for name, frame, audit in [('original_control', control, ca), ('uniform_candidate', candidate, ea)]:
        assert frame.columns.tolist() == ['StudyInstanceUID', *targets]
        pred = frame.set_index('StudyInstanceUID').loc[ids, targets].to_numpy()
        assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
        summary = summarize(truth, baseline, pred, targets, .10)
        assert abs(summary['candidate_auc'] - audit['expert_macro_auc']) < 1e-9
        rank = pd.DataFrame(pred).rank(method='average', pct=True).to_numpy()
        mixture = .9 * base_rank + .1 * rank
        interval, valid = paired_interval(truth, base_rank, mixture, groups)
        summary.update(paired_group_interval=interval, valid_bootstrap_draws=valid)
        keep = np.array([rows[u]['fold'] != 2 for u in ids])
        summary['sensitivity_excluding_fold2'] = {
            'studies': int(keep.sum()),
            'mix_delta': float(roc_auc_score(truth[keep], mixture[keep]) - roc_auc_score(truth[keep], base_rank[keep])),
            'ranks': 'Full58 cohort ranks retained when excluding fold2',
        }
        arms[name] = summary
    result = {'status': 'COMPLETE_FIXED_WEIGHT_COMPLEMENTARITY_DIAGNOSTIC', 'arms': arms,
              'protocol_sha256': sha(PROTOCOL), 'weight_search_performed': False,
              'control_audit_sha256': sha(args.control_audit), 'candidate_audit_sha256': sha(args.candidate_audit),
              'competition_submission_enabled': False, 'public_score_prediction': None,
              'decision': 'REVIEW_EVIDENCE_AND_SERVING_PARITY_BEFORE_ANY_PROMOTION',
              'limitations': protocol['limitations']}
    (RUN / 'complete_blend_result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))


if __name__ == '__main__':
    main()
