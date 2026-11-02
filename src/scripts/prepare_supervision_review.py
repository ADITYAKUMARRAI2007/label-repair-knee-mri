#!/usr/bin/env python3
"""Prepare a reproducible report review without fitting to expert target values."""
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/supervision_review_v1'
SEED = 20261008
PRIORITY = ['Synovitis', 'Fracture', 'PF OA', 'Medial Meniscus', 'Lateral Meniscus']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        records = list(csv.DictReader(handle))
    ids = [r['StudyInstanceUID'] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f'duplicate studies: {path.name}')
    return {r['StudyInstanceUID']: r for r in records}


def wording_hint(report):
    # Vocabulary buckets for review stratification, not validated language IDs.
    tokens = set(re.findall(r'\w+', report.lower()))
    vocabulary = {
        'german_markers': {'kein', 'keine', 'knie', 'befund', 'ohne', 'und', 'des'},
        'italian_markers': {'ginocchio', 'della', 'del', 'non', 'lesione', 'liquido'},
        'portuguese_markers': {'joelho', 'sem', 'nao', 'não', 'lesao', 'lesão'},
        'spanish_markers': {'rodilla', 'sin', 'lesión', 'rotura', 'del'},
        'english_markers': {'knee', 'the', 'tear', 'there', 'with', 'without'},
    }
    scores = {name: len(tokens & words) for name, words in vocabulary.items()}
    best = max(scores.values())
    winners = [name for name, count in scores.items() if count == best]
    return winners[0] if best >= 2 and len(winners) == 1 else 'unresolved'


def main():
    freeze = ROOT / 'runs/E04_contract/e04_contract.json'
    contract = json.loads(freeze.read_text())
    targets = contract['target_order']
    official_path = ROOT / 'data/train.csv'
    if digest(official_path) != contract['official_train_sha256']:
        raise RuntimeError('official input changed')
    official = read_csv(official_path)
    meta = {r['StudyInstanceUID']: r for r in contract['rows']}
    if set(official) != set(meta):
        raise RuntimeError('study roster changed')
    gold_groups = {r['report_group'] for r in meta.values() if r['expert']}
    eligible = sorted(u for u, r in meta.items() if not r['expert'] and r['report_group'] not in gold_groups)
    excluded = sorted(set(meta) - set(eligible))
    # Access expert membership only. No expert finding values are evaluated.
    sources, hashes = {}, {str(freeze.relative_to(ROOT)): digest(freeze),
                           str(official_path.relative_to(ROOT)): digest(official_path)}
    for name, receipt in contract['fusion']['source_receipts'].items():
        path = ROOT / receipt['path']
        if digest(path) != receipt['sha256']:
            raise RuntimeError(f'label source changed: {name}')
        sources[name] = read_csv(path)
        if not set(eligible) <= set(sources[name]):
            raise RuntimeError(f'weak-label coverage missing: {name}')
        hashes[str(path.relative_to(ROOT))] = digest(path)
    pilk = sources['report_labels_v2.csv']
    cell = {}
    for u in eligible:
        for t in targets:
            values = [float(rows[u][t]) for rows in sources.values()]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise RuntimeError('invalid source probability')
            cell[u, t] = {'source_probabilities': dict(zip(sources, values)),
                          'median': statistics.median(values), 'range': max(values)-min(values),
                          'verdict': pilk[u][t+'__verdict'], 'source_confidence': float(pilk[u][t+'__conf'])}
    group_members = defaultdict(list)
    for u in eligible:
        group_members[meta[u]['report_group']].append(u)
    groups = sorted(group_members)
    rng = random.Random(SEED)
    holdout_groups = set(rng.sample(groups, 40))
    pool = [g for g in groups if g not in holdout_groups]
    random_groups = rng.sample(pool, 50)
    chosen = set(random_groups)
    targeted = []
    # Ten distinct report groups per prespecified finding, independent of Gold.
    for t in PRIORITY:
        ranking = sorted((g for g in pool if g not in chosen),
                         key=lambda g: (-max(cell[u,t]['range'] for u in group_members[g]), g))
        pick = ranking[:10]
        targeted.extend((g,t) for g in pick)
        chosen.update(pick)
    development = []
    for g in random_groups:
        u = group_members[g][0]
        t = max(targets, key=lambda finding: cell[u,finding]['range'])
        development.append((u, 'random', t))
    for g,t in targeted:
        u = sorted(group_members[g], key=lambda uid: (-cell[uid,t]['range'],uid))[0]
        development.append((u, 'targeted_disagreement', t))
    holdout = [group_members[g][0] for g in sorted(holdout_groups)]
    dev_uids = [u for u,_,_ in development]
    assert len(dev_uids) == len(set(dev_uids)) == 100
    assert len(holdout) == 40 and not set(dev_uids) & set(holdout)
    assert not chosen & holdout_groups and not (chosen | holdout_groups) & gold_groups
    split = {'schema': 'supervision_report_review_split_v1', 'seed': SEED,
             'development_ids': dev_uids, 'development_report_groups': sorted(chosen),
             'holdout_ids': holdout, 'holdout_report_groups': sorted(holdout_groups),
             'excluded_expert_and_connected_ids': excluded,
             'input_sha256': hashes,
             'holdout_rule': '40 random report groups untouched during rule development',
             'expert_target_values_used': False, 'patient_site_separation_verified': False}
    OUT.mkdir(parents=True, exist_ok=True)
    split_path = OUT / 'split.json'
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise RuntimeError('existing frozen report sample would change')
    split_path.write_text(json.dumps(split,indent=2)+'\n')
    coverage = []
    for t in targets:
        verdicts = Counter(cell[u,t]['verdict'] for u in eligible)
        unknown = [cell[u,t]['median'] for u in eligible if cell[u,t]['verdict'] == 'UNK']
        coverage.append({'target':t,'studies':len(eligible),'YES':verdicts['YES'],
                         'NO':verdicts['NO'],'UNK':verdicts['UNK'],
                         'mean_source_range':sum(cell[u,t]['range'] for u in eligible)/len(eligible),
                         'source_range_ge_0_5':sum(cell[u,t]['range']>=.5 for u in eligible),
                         'unknown_median_consensus':statistics.median(unknown) if unknown else None})
    with (OUT/'coverage.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    pack = [{'review_index':i+1,'StudyInstanceUID':u,'sampling':kind,'focus_target':t,
             'wording_hint':wording_hint(official[u]['Report']),
             'Report':official[u]['Report'],'focus_labels':cell[u,t],
             'all_target_labels':{target:cell[u,target] for target in targets}}
            for i,(u,kind,t) in enumerate(development)]
    (OUT/'development_reports.json').write_text(json.dumps(pack,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'review_annotations.csv').open('x',newline='') as handle:
        fields=['review_index','StudyInstanceUID','target','report_evidence','report_state',
                'source_extraction_issue','reviewer','notes']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in pack:
            writer.writerow({'review_index':r['review_index'],'StudyInstanceUID':r['StudyInstanceUID'],
                             'target':r['focus_target']})
    summary={'schema':'supervision_review_preparation_v1','status':'PREPARED_NOT_REVIEWED',
             'observed_utc':datetime.now(timezone.utc).isoformat(),
             'eligible_studies':len(eligible),'excluded_expert_and_connected':len(excluded),
             'development_reports':len(pack),'untouched_holdout_groups':len(holdout),
             'wording_hint_counts':dict(Counter(wording_hint(official[u]['Report']) for u in eligible)),
             'limitations':['verdicts are extractor outputs, not image truth',
                            'wording buckets are unvalidated language hints',
                            'source agreement can share correlated errors',
                            'no validated patient/site grouping'],
             'script_sha256':digest(Path(__file__)),
             'artifacts':{f.name:digest(f) for f in OUT.iterdir() if f.is_file() and f.name!='preparation.json'},
             'label_policy_changed':False,'gpu_jobs_launched':0,'submissions_launched':0}
    (OUT/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('artifacts','limitations')},indent=2))


if __name__ == '__main__':
    main()
