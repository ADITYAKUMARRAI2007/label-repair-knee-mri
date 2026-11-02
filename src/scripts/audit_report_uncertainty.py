#!/usr/bin/env python3
"""Audit existing weak supervision without reading expert finding values.

UNK is a labeler's unmentioned-or-uncertain verdict, not verified image truth.
No labels are changed and no models or external APIs are executed.
"""
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/report_uncertainty_audit_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {r['StudyInstanceUID']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate study IDs in {path.name}')
    return result


def summary(values):
    if not values:
        return {'n': 0}
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values),
            'max': max(values)}


def main():
    contract_path = ROOT / 'runs/E04_contract/e04_contract.json'
    split_path = ROOT / 'runs/E03_contract/split.json'
    contract = json.loads(contract_path.read_text())
    split = json.loads(split_path.read_text())
    excluded = set(split['gold']) | set(split['excluded_gold_group_weak'])
    ids = sorted(r['StudyInstanceUID'] for r in contract['rows']
                 if r['StudyInstanceUID'] not in excluded)
    id_set = set(ids)
    assert len(ids) == 4345 and len(excluded) == 62
    assert all(not r['expert'] for r in contract['rows']
               if r['StudyInstanceUID'] in id_set)
    targets = contract['target_order']
    sources, hashes = {}, {str(p.relative_to(ROOT)): sha(p)
                            for p in [contract_path, split_path]}
    for name, receipt in contract['fusion']['source_receipts'].items():
        path = ROOT / receipt['path']
        hashes[receipt['path']] = sha(path)
        assert hashes[receipt['path']] == receipt['sha256']
        sources[name] = read_rows(path)
        assert set(ids) <= sources[name].keys()
    source_names = list(sources)
    primary = sources['report_labels_v2.csv']
    control = read_rows(ROOT / 'runs/grade_supervision_trial_v1/control_targets.csv')
    weights = read_rows(ROOT / 'runs/grade_supervision_trial_v1/shared_weights.csv')
    for name in ['control_targets.csv', 'shared_weights.csv']:
        path = ROOT / 'runs/grade_supervision_trial_v1' / name
        hashes[str(path.relative_to(ROOT))] = sha(path)
    cells = []
    max_target_error = max_weight_error = 0.
    for uid in ids:
        for target in targets:
            values = [float(sources[n][uid][target]) for n in source_names]
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            probability = min(.99, max(.01, statistics.median(values)))
            weight = min(1., max(.25, 1 - (max(values) - min(values))))
            verdict = primary[uid][target + '__verdict']
            confidence = float(primary[uid][target + '__conf'])
            assert verdict in {'YES', 'NO', 'UNK'} and 0 <= confidence <= 1
            max_target_error = max(max_target_error, abs(probability - float(control[uid][target])))
            max_weight_error = max(max_weight_error, abs(weight - float(weights[uid][target])))
            cells.append({'target': target, 'verdict': verdict, 'confidence': confidence,
                          'p': probability, 'weight': weight, 'values': values})
    assert max(max_target_error, max_weight_error) < 1e-6
    by_target = {}
    for target in targets + ['ALL']:
        selected = [c for c in cells if target == 'ALL' or c['target'] == target]
        states = {}
        for verdict in ['YES', 'NO', 'UNK']:
            group = [c for c in selected if c['verdict'] == verdict]
            states[verdict] = {
                'count': len(group),
                'fraction': len(group) / len(selected),
                'fused_probability': summary([c['p'] for c in group]),
                'current_weight': summary([c['weight'] for c in group]),
                'primary_confidence': summary([c['confidence'] for c in group]),
                'probability_le_010': sum(c['p'] <= .1 for c in group),
                'probability_ge_050': sum(c['p'] >= .5 for c in group),
                'weight_ge_070': sum(c['weight'] >= .7 for c in group),
                'most_common_probabilities': Counter(round(c['p'], 6) for c in group).most_common(8),
                'source_means': {name: statistics.mean(c['values'][i] for c in group)
                                 for i, name in enumerate(source_names)} if group else {},
            }
        by_target[target] = states
    pairs = {}
    for a, b in itertools.combinations(range(len(source_names)), 2):
        av = [c['values'][a] for c in cells]
        bv = [c['values'][b] for c in cells]
        pairs[f'{source_names[a]} | {source_names[b]}'] = {
            'pooled_pearson': statistics.correlation(av, bv),
            'mean_absolute_difference': statistics.mean(abs(x-y) for x,y in zip(av,bv)),
            'threshold_05_agreement': statistics.mean((x >= .5) == (y >= .5) for x,y in zip(av,bv)),
        }
    unknown = [c for c in cells if c['verdict'] == 'UNK']
    report = {
        'schema': 'report_uncertainty_audit_v1',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'training_studies': len(ids), 'excluded_studies': len(excluded),
        'training_cells': len(cells), 'expert_finding_values_used': False,
        'source_hashes': hashes, 'script_sha256': sha(Path(__file__)),
        'control_reconstruction_max_error': {'targets': max_target_error, 'weights': max_weight_error},
        'by_target': by_target, 'pairwise_sources': pairs,
        'unknown_share_of_total_weight': sum(c['weight'] for c in unknown) / sum(c['weight'] for c in cells),
        'limitations': [
            'UNK combines report silence and textual uncertainty; neither proves an image finding absent or present.',
            'One upstream labeler supplies verdicts; errors and correlated label sources remain possible.',
            'Correlations pool targets and are descriptive, not proof of shared source lineage.',
            'Expert studies and report-linked rows are excluded, but upstream prompt/label calibration independence is unverified.',
            'No English-only keyword absence heuristic or new medical labeling is used.',
            'No training, validation AUC, label change, or submission is performed by this audit.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'training_cells': len(cells), 'ALL': by_target['ALL'],
                      'unknown_share_of_total_weight': report['unknown_share_of_total_weight'],
                      'control_reconstruction_max_error': report['control_reconstruction_max_error']}, indent=2))


if __name__ == '__main__':
    main()
