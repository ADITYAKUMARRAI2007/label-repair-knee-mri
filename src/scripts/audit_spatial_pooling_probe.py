#!/usr/bin/env python3
"""Independently audit a completed fixed paired probe; never submit a model."""
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
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--attempt',type=int,choices=[1,2],default=1)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / 'manifest.json').read_text())
    suffix='' if args.attempt==1 else '_v2'
    build = json.loads((ROOT / f'runs/spatial_pooling_probe_source{suffix}.json').read_text())
    require(manifest['status'] == 'COMPLETE', 'incomplete comparison')
    require(manifest['contract'] == build['contract'], 'frozen contract differs')
    require(not manifest['competition_submission_enabled'] and manifest['public_score'] is None,
            'unexpected public scoring')
    for relative, expected in {**build['sources'], **build['gates']}.items():
        require(sha(ROOT / relative) == expected, f'frozen source changed: {relative}')
    for name, expected in manifest['artifacts'].items():
        require(Path(name).name == name, 'invalid artifact name')
        path = directory / name
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
                f'artifact mismatch: {name}')
    require(manifest['global_feature_reference_exact'], 'global reference parity failed')
    require(manifest['resource_pilot']['status'] == 'PASS' and manifest['resource_pilot']['pilot_expert_rows'] == 0, 'resource pilot failed')
    require(manifest['feature_receipts']['candidate']['shape'] == [4407,20,49,2048], 'full grid not used')
    require(all(not r['persisted'] for r in manifest['feature_receipts'].values()), 'unexpected feature persistence')
    contract = build['contract']
    if args.attempt==2:
        resource=manifest['resource_pilot']
        estimates={str(d):sum(resource['pilot'][str(d)][arm]['p95_complete_step_seconds']*68*30 for arm in ['control','candidate'])*len(contract['seeds'][d::2])*1.2 for d in range(2)}
        require(all(abs(estimates[d]-resource['projected_device_head_seconds'][d])<1e-8 for d in estimates),'device projection math differs')
        require(abs(max(estimates.values())-resource['projected_all_head_seconds'])<1e-8,'full fit projection differs')
        require(resource['projected_worker_elapsed_seconds']<3300,'complete fixed recipe exceeds cap')
    train_path = ROOT / 'data/train.csv'
    require(sha(train_path) == contract['train_sha256'], 'official data changed')
    with train_path.open(newline='') as handle:
        rows = {r['StudyInstanceUID']: r for r in csv.DictReader(handle)}
    split = json.loads((ROOT / 'runs/E03_contract/split.json').read_text())
    ids = np.load(directory / 'expert_ids.npy', allow_pickle=False).astype(str)
    require(len(ids) == 58 and len(set(ids)) == 58 and set(ids) == set(split['gold']),
            'expert cohort differs')
    train_ids = sorted(set(rows) - set(split['gold']) - set(split['excluded_gold_group_weak']))
    require(hashlib.sha256(json.dumps(train_ids).encode()).hexdigest() == contract['train_ids_sha256'],
            'training exclusion drift')
    truth = np.asarray([[float(rows[uid][t]) for t in contract['targets']] for uid in ids])
    require(np.isin(truth, [0, 1]).all(), 'expert truth is not binary')
    results = {(r['seed'], r['arm']): r for r in manifest['results']}
    require(len(results) == 6 and len(manifest['results']) == 6, 'six fixed fits required')
    scores = {arm: [] for arm in ['control', 'candidate']}
    predictions = {}
    for seed in contract['seeds']:
        require(results[seed, 'control']['initial_state_sha256'] == results[seed, 'candidate']['initial_state_sha256'],
                'paired initialization differs')
        for arm in scores:
            result = results[seed, arm]
            if args.attempt==2:
                require(result['physical_device']==contract['seeds'].index(seed)%2,'seed GPU assignment differs')
            require(result['epochs'] == 30 and [r['epoch'] for r in result['history']] == list(range(1, 31)),
                    'fixed schedule incomplete')
            require(np.isfinite([r['loss'] for r in result['history']]).all(), 'invalid loss')
            require(result['checkpoint_replay_max_abs_error'] <= 1e-6, 'checkpoint replay failed')
            prediction = np.load(directory / f'{arm}_{seed}_expert.npy', allow_pickle=False)
            require(prediction.shape == (58, 12) and np.isfinite(prediction).all()
                    and ((prediction >= 0) & (prediction <= 1)).all(), 'invalid predictions')
            auc = np.asarray([roc_auc_score(truth[:, j], prediction[:, j]) for j in range(12)])
            require(np.allclose(auc, [result['expert_per_target_auc'][t] for t in contract['targets']], atol=1e-12, rtol=0),
                    'reported target AUC differs')
            require(abs(auc.mean() - result['expert_macro_auc']) < 1e-12, 'reported macro differs')
            predictions[seed, arm] = prediction
            scores[arm].append(auc)
    control, candidate = np.asarray(scores['control']), np.asarray(scores['candidate'])
    delta = candidate - control
    # One fixed bootstrap, paired by study and shared across all seeds/targets.
    counts = np.random.default_rng(20261010).multinomial(58, np.full(58, 1 / 58), size=2000)
    boot_delta = np.zeros(2000)
    valid = np.ones(2000, bool)
    for j in range(12):
        pos, neg = truth[:, j] == 1, truth[:, j] == 0
        positive, negative = counts[:, pos], counts[:, neg]
        denominator = positive.sum(1) * negative.sum(1)
        valid &= denominator > 0
        for seed in contract['seeds']:
            pair_delta = np.zeros((pos.sum(), neg.sum()))
            for arm, sign in [('candidate', 1), ('control', -1)]:
                pred = predictions[seed, arm][:, j]
                diff = pred[pos, None] - pred[None, neg]
                pair_delta += sign * ((diff > 0).astype(float) + .5 * (diff == 0))
            numerator = np.einsum('bi,ij,bj->b', positive, pair_delta, negative, optimize=True)
            boot_delta += np.divide(numerator, denominator, out=np.zeros(2000), where=denominator > 0) / 36
    require(valid.sum() >= 1900, 'too many class-degenerate bootstrap samples')
    lower, upper = np.quantile(boot_delta[valid], [.025, .975])
    targeted = [contract['targets'].index(t) for t in contract['trial']['targeted_findings']]
    gates = {'candidate_macro_at_least_090': float(candidate.mean()) >= .90,
             'macro_gain_at_least_0003': float(delta.mean()) >= .003,
             'targeted_gain_at_least_001': float(delta[:, targeted].mean()) >= .01,
             'positive_macro_in_two_seeds': int((delta.mean(1) > 0).sum()) >= 2,
             'paired_bootstrap_lower_above_zero': float(lower) > 0}
    report = {'status': 'PASS_SCREENING_ONLY' if all(gates.values()) else 'REJECT_OR_INCONCLUSIVE',
              'control_mean_seed_macro': float(control.mean()), 'candidate_mean_seed_macro': float(candidate.mean()),
              'mean_seed_macro_gain': float(delta.mean()), 'targeted_mean_seed_gain': float(delta[:, targeted].mean()),
              'per_seed_macro_gains': delta.mean(1).tolist(), 'gates': gates,
              'per_target_mean_seed_gain': dict(zip(contract['targets'], delta.mean(0).tolist())),
              'paired_study_bootstrap_95pct': [float(lower), float(upper)],
              'bootstrap_valid_samples': int(valid.sum()), 'public_score': None,
              'eligible_for_competition_submission': False,
              'limitations': contract['trial']['limitations'],
              'feature_hashes_verified_locally': False,
              'feature_hash_note': 'Temporary remote feature cache hashes are producer receipts; persisted predictions/checkpoints are independently hashed.',
              'remaining_gates': ['full candidate training', 'raw DICOM parity', 'runtime', 'official standalone score']}
    (directory / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
