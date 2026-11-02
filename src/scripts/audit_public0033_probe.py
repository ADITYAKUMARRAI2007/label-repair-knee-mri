#!/usr/bin/env python3
"""Audit fixed specialist outputs before considering another action."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len({r['StudyInstanceUID'] for r in rows}) == len(rows), 'duplicate identities')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    directory = parser.parse_args().directory
    manifest = json.loads((directory / 'manifest.json').read_text())
    build = json.loads((ROOT / 'runs/public0033_probe_source.json').read_text())
    contract = build['contract']
    require(manifest['status'] == 'COMPLETE' and manifest['contract'] == contract, 'contract or completion differs')
    for relative, expected in {**build['sources'], **build['gates']}.items():
        require(sha(ROOT / relative) == expected, 'source/gate changed: ' + relative)
    for name, expected in manifest['artifacts'].items():
        require(Path(name).name == name, 'invalid output path')
        p = directory / name
        require(p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256'], 'artifact differs: ' + name)
    require(manifest['fallback'] == 0 and not manifest['competition_submission_enabled'] and manifest['public_score'] is None,
            'unexpected fallback or submission')
    require(manifest['reference_parity']['status'] == 'PASS', 'reference parity failed')
    original = {r['StudyInstanceUID']: r for r in contract['reference_predictions']}
    reference = read(directory / 'reference_predictions.csv')
    require({r['StudyInstanceUID'] for r in reference} == set(original), 'reference cohort differs')
    errors = [abs(float(row[t]) - float(original[row['StudyInstanceUID']][t]))
              for row in reference for t in ['Medial Meniscus', 'Lateral Meniscus']]
    require(max(errors) <= contract['reference_tolerance'], 'recomputed reference error too large')
    expert = read(directory / 'expert_predictions.csv')
    require(len(expert) == 58 and {r['StudyInstanceUID'] for r in expert} == set(contract['expert_ids']),
            'expert cohort differs')
    train_path = ROOT / 'data/train.csv'
    require(sha(train_path) == contract['official_file_sha256']['train.csv'], 'official truth changed')
    official = {r['StudyInstanceUID']: r for r in read(train_path)}
    ids = [r['StudyInstanceUID'] for r in expert]
    scores = np.asarray([[float(r[t]) for t in contract['targets']] for r in expert])
    truth = np.asarray([[float(official[uid][t]) for t in contract['targets']] for uid in ids])
    require(scores.shape == (58,12) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(), 'invalid predictions')
    require(np.isin(truth,[0,1]).all(), 'expert truth not binary')
    cohorts = {}
    for name, keep in [('all58_known_report_connection', np.ones(58,bool)),
                       ('57_without_known_report_connections', np.asarray([uid not in contract['report_connected_expert_ids'] for uid in ids]))]:
        auc = {t:float(roc_auc_score(truth[keep,j], scores[keep,j])) for j,t in enumerate(contract['targets'])}
        cohorts[name] = {'studies':int(keep.sum()), 'per_target_auc':auc, 'macro_auc':float(np.mean(list(auc.values())))}
    require(cohorts['57_without_known_report_connections']['studies']==57, 'report-link partition drift')
    macro = cohorts['all58_known_report_connection']['macro_auc']
    report = {'status':'REJECT_BELOW_MACRO_FLOOR' if macro < .90 else 'PASS_COARSE_FLOOR_REQUIRES_COMPLEMENTARITY_REVIEW',
              'cohorts':cohorts, 'reference_max_abs_error':max(errors), 'public_score':None,
              'eligible_for_competition_submission':False, 'validation_limitations':contract['validation_limitations'],
              'remaining_gates':['complementarity review','full serving runtime','official standalone score before any blend']}
    (directory / 'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
