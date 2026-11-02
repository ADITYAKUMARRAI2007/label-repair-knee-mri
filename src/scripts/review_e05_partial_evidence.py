"""Review completed E05 folds for further compute; never promote partial OOF."""
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_e05_complete_oof import ROOT, paired_interval, sha

RUN = ROOT / 'runs/e05_confidence_ablation'
PAIR = ROOT / 'runs/e05_kaggle_pair4'


def audit(directories, experiment, output, reuse=False):
    command = [sys.executable, str(ROOT / 'scripts/audit_e04_fold_training.py'),
               *map(str, directories), '--experiment', experiment, '--output', str(output)]
    if reuse:
        command.append('--reuse-e06-control-fold2')
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=120)
    return json.loads(output.read_text())


def collect(directories, prefix, audit_record):
    result = {}
    audited = {f['fold']: f for f in audit_record['folds']}
    for directory in directories:
        for manifest_path in directory.rglob(f'{prefix}_fold[0-4]_manifest.json'):
            manifest = json.loads(manifest_path.read_text())
            fold = manifest['fold']
            assert sha(manifest_path) == audited[fold]['manifest_sha256']
            assert fold not in result
            path = manifest_path.parent / f'{prefix}_fold{fold}_oof_predictions.csv'
            receipt = manifest['artifacts'][path.name]
            assert sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes']
            result[fold] = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    return result


def main():
    control_dirs = [ROOT / 'runs/E04_folds_01_outputs', ROOT / 'runs/E04_folds_23_outputs']
    candidate_dirs = [ROOT / f'runs/e05_full_fold{f}/verified_final' for f in [0, 1]]
    include_four = False
    watcher_path = PAIR / 'watcher.json'
    if watcher_path.exists():
        watcher = json.loads(watcher_path.read_text())
        include_four = (watcher.get('status') == 'FINISHED'
                        and watcher.get('result') == 'BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED')
    if include_four:
        control_dirs.append(PAIR / 'output/e04_fold4')
        candidate_dirs.append(PAIR / 'output/e05_uniform_fold4')
    control_audit_path = RUN / 'compute_review_control_audit.json'
    candidate_audit_path = RUN / 'compute_review_candidate_audit.json'
    ca = audit(control_dirs, 'e04', control_audit_path)
    ea = audit(candidate_dirs, 'e05_uniform', candidate_audit_path, reuse=True)
    control = collect(control_dirs, 'e04', ca)
    candidate = collect(candidate_dirs, 'e05_uniform', ea)
    amendment_path = RUN / 'e06_control_reuse_amendment.json'
    amendment = json.loads(amendment_path.read_text())
    reused_dir = ROOT / amendment['artifact_directory']
    reused_audit_path = RUN / 'compute_review_reused_fold_audit.json'
    ra = audit([reused_dir], 'e06_control', reused_audit_path)
    candidate.update(collect([reused_dir], 'e06_control', ra))
    folds = sorted(candidate)
    assert folds == ([0, 1, 2, 4] if include_four else [0, 1, 2])
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    targets = frozen['target_order']
    rows = {r['StudyInstanceUID']: r for r in frozen['rows'] if r['expert']}
    ids = sorted(u for u, r in rows.items() if r['fold'] in folds)
    # Fold sizes come from the audited split; the final folds have11 experts,
    # rather than an assumed10or12.
    assert len(ids) == sum(f['expert_rows'] for f in ea['folds'])
    arrays = {}
    for name, frames in [('control', control), ('candidate', candidate)]:
        frame = pd.concat([frames[f] for f in folds], ignore_index=True)
        assert frame.StudyInstanceUID.is_unique
        assert frame.columns.tolist() == ['StudyInstanceUID', *targets]
        arrays[name] = frame.set_index('StudyInstanceUID').loc[ids, targets].to_numpy()
    blend_protocol_path = RUN / 'complete_blend_protocol.json'
    blend_protocol = json.loads(blend_protocol_path.read_text())
    assert blend_protocol['candidate_weight'] == .1 and blend_protocol['baseline_weight'] == .9
    baseline_path = ROOT / blend_protocol['baseline_path']
    assert sha(baseline_path) == blend_protocol['baseline_sha256']
    arrays['baseline'] = pd.read_csv(baseline_path, dtype={'StudyInstanceUID': str}).set_index(
        'StudyInstanceUID').loc[ids, targets].to_numpy()
    y = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index(
        'StudyInstanceUID').loc[ids, targets].to_numpy()
    groups = np.array([rows[u]['report_group'] for u in ids])
    for pred in arrays.values():
        assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    ranks = {k: pd.DataFrame(v).rank(method='average', pct=True).to_numpy()
             for k, v in arrays.items()}
    results = {}
    for name in ['control', 'candidate']:
        mix = .9 * ranks['baseline'] + .1 * ranks[name]
        per_target = roc_auc_score(y, mix, average=None)
        interval, draws = paired_interval(y, ranks['baseline'], mix, groups)
        base_auc = float(roc_auc_score(y, ranks['baseline']))
        results[name] = {
            'standalone_auc': float(roc_auc_score(y, arrays[name])),
            'baseline_auc': base_auc, 'fixed_blend_auc': float(per_target.mean()),
            'fixed_blend_delta': float(per_target.mean()) - base_auc,
            'paired_group_interval': interval, 'valid_bootstrap_draws': draws,
            'per_target_blend_delta': dict(zip(targets, (
                per_target - roc_auc_score(y, ranks['baseline'], average=None)).tolist())),
        }
    raw_interval, draws = paired_interval(y, arrays['control'], arrays['candidate'], groups)
    keep = np.array([rows[u]['fold'] != 2 for u in ids])
    result = {
        'status': 'PARTIAL_COMPUTE_REVIEW_ONLY',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'folds': folds, 'expert_studies': len(ids), 'fold4_included': include_four,
        'missing_candidate_folds': sorted(set(range(5)) - set(folds)),
        'standalone_delta': results['candidate']['standalone_auc'] - results['control']['standalone_auc'],
        'standalone_paired_group_interval': raw_interval, 'valid_bootstrap_draws': draws,
        'fixed_blend_results': results,
        'sensitivity_excluding_reused_fold2': {
            'studies': int(keep.sum()),
            'standalone_delta': float(roc_auc_score(y[keep], arrays['candidate'][keep])
                                      - roc_auc_score(y[keep], arrays['control'][keep])),
        },
        'source_sha256': sha(ROOT / 'scripts/review_e05_partial_evidence.py'),
        'audits_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                          [control_audit_path, candidate_audit_path, reused_audit_path]},
        'baseline_sha256': sha(baseline_path),
        'blend_protocol_sha256': sha(blend_protocol_path),
        'reuse_amendment_sha256': sha(amendment_path),
        'next_action': 'REVIEW_COMPLETED_FOLD4_BEFORE_ANY_MORE_TRAINING',
        'additional_gpu_launched': False, 'competition_submission_enabled': False,
        'complete_oof': False, 'public_score_prediction': None,
        'limitations': [
            'Partial analysis was added after folds0/1/2 results were known; it is a compute review.',
            'Gold labels have been reused; this cannot establish an independent generalization gain.',
            'Fold2 reuses compatible E06 control training; hardware and implementation differ.',
            'Ranks use only the matched partial cohort, not58 experts or full competition serving data.',
            'Bootstrap uncertainty is conditional on fixed predictions, not training variability.',
            'Only complete five-fold evidence can support the existing promotion review.',
        ],
    }
    output = RUN / 'partial_compute_review.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['audits_sha256', 'limitations']}))


if __name__ == '__main__':
    main()
