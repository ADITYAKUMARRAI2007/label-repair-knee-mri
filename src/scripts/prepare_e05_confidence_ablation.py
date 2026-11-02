"""Freeze a controlled training comparison; do not launch or claim accuracy."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'training/e05_confidence.py'
    spec = importlib.util.spec_from_file_location('confidence', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract_path = ROOT/'runs/E04_contract/e04_contract.json'
    contract = json.loads(contract_path.read_text())
    for name, expected in contract['arrays'].items():
        assert sha(contract_path.parent/name) == expected
    weights = np.load(contract_path.parent/'e04_weights.npy')
    records = contract['rows']
    expert = np.array([r['expert'] for r in records])
    folds = np.array([r['fold'] for r in records])
    groups = np.array([r['report_group'] for r in records])
    checks = []
    for fold in range(5):
        rows = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        assert not set(groups[rows]) & set(groups[held])
        result, means = module.uniform_weak_weights(weights, expert, rows)
        weak_rows = rows[~expert[rows]]
        np.testing.assert_allclose(result[weak_rows].sum(0, dtype=np.float64),
                                   weights[weak_rows].sum(0, dtype=np.float64), rtol=1e-6)
        np.testing.assert_array_equal(result[expert], weights[expert])
        np.testing.assert_array_equal(result[held], weights[held])
        perturbed = weights.copy()
        perturbed[held] *= 17
        second, second_means = module.uniform_weak_weights(perturbed, expert, rows)
        np.testing.assert_array_equal(means, second_means)
        np.testing.assert_array_equal(result[rows], second[rows])
        checks.append({'fold': fold, 'training_rows': len(rows),
                       'expert_held_out': int(expert[held].sum()),
                       'weight_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
                       'per_target_weak_weight': means.tolist()})
    protocol = {
        'status': 'SPECIFICATION_AND_WEIGHT_TRANSFORM_READY_NOT_LAUNCHED',
        'question': 'Does source-disagreement weighting harm generalization compared with equal weak-example weights?',
        'control': 'Exact original E04 confidence weights',
        'candidate': 'Within each training fold, equal weak weights per target at that fold training weak mean; expert weight unchanged',
        'unchanged': ['Generic pretrained checkpoint', 'Label probabilities', 'Grouped folds',
                      'Input cache', 'Augmentation', 'Architecture', 'Learning rates',
                      'Optimizer', 'Matched initialization and batch-order seeds',
                      '14 epochs', 'Final EMA checkpoint'],
        'required_execution': ['Build separate paired trainer and resource pilot',
                               'Measure full epoch time including I/O on available hardware',
                               'Check fresh available compute before full-training commitment',
                               'Train fresh separate control/candidate backbones; do not warm-start from expert-trained other folds',
                               'Complete five-fold held-out expert predictions before a performance decision'],
        'selection': 'One fixed comparison; no target-specific routing or weight search',
        'validation': 'Same grouped folds prevent new within-run report-group leakage, but Gold58 is reused and remains a biased diagnostic',
        'promotion': 'An owned model needs verified stronger held-out performance and complementary errors before any competition scoring decision',
        'risk': 'Disagreement may flag actual annotation errors; removing its weighting can worsen the model. No score gain is assumed.',
        'expert_labels_used_to_set_candidate_weights': False,
        'gpu_job_started': False, 'competition_submission_enabled': False,
        'source_sha256': sha(source), 'e04_contract_sha256': sha(contract_path),
        'checks': checks,
    }
    out = ROOT/'runs/e05_confidence_ablation';out.mkdir(exist_ok=True)
    target = out/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == protocol, 'Preserve previously frozen protocol'
    else:
        target.write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'status': protocol['status'], 'fold_checks': len(checks),
                      'training_mass_preserved': True, 'heldout_mutation_invariance': True,
                      'protocol_sha256': sha(target)}))


if __name__ == '__main__':
    main()
