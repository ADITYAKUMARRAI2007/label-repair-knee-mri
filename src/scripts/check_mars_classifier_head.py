"""Check the actual classifier on real frozen features; no accuracy claim."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from mars_spatial_head import MarsSpatialHead
from mars_classifier_train import split_rows


def main():
    torch.set_num_threads(2); torch.manual_seed(20261010)
    run = ROOT / 'research/foundation_refresh_20261010/MARS'
    protocol_path = run / 'classifier_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol['sources_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    fixture = run / 'real_study_cpu_fixture.npz'
    with np.load(fixture, allow_pickle=False) as a:
        features = torch.from_numpy(a['features'].copy())
        present = torch.from_numpy(a['present'][None].copy())
    assert features.shape == (1, 3, 768, 3, 3, 3)
    model = MarsSpatialHead().eval()
    with torch.no_grad():
        reference = model(features, present)
        assert reference.shape == (1, 12) and torch.isfinite(reference).all()
        pair = torch.cat([features, features * 1.1 + .2])
        masks = present.repeat(2, 1)
        forward = model(pair, masks)
        reverse = model(pair.flip(0), masks.flip(0)).flip(0)
        permutation_error = float((forward - reverse).abs().max())
        assert permutation_error < 1e-6
        missing = present.clone()
        missing[0, int(present[0].nonzero()[0])] = False
        assert missing.any()
        zeroed = torch.where(missing[:, :, None, None, None, None], features, 0.)
        corrupted = zeroed.clone(); corrupted[~missing] = float('nan')
        missing_error = float((model(zeroed, missing) - model(corrupted, missing)).abs().max())
        assert missing_error == 0.
        try:
            model(features, torch.zeros_like(present))
        except ValueError:
            pass
        else:
            raise AssertionError('All-missing study accepted')
    model.train()
    loss = model(features, present).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    rows = frozen['rows']; fold_counts = []
    covered = set()
    for fold in range(5):
        training, validation = split_rows(rows, fold)
        assert not set(training) & set(validation)
        assert not set(r['report_group'] for i, r in enumerate(rows) if i in set(training)) & set(
            rows[i]['report_group'] for i in validation)
        assert not covered & set(validation)
        covered.update(validation)
        fold_counts.append({'fold': fold, 'training': len(training), 'heldout': len(validation),
                            'expert_heldout': sum(rows[i]['expert'] for i in validation)})
    assert covered == set(range(4407)) and sum(x['expert_heldout'] for x in fold_counts) == 58
    result = {'status': 'PASS_REAL_FEATURE_HEAD_AND_GROUP_SPLIT_CHECKS',
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'real_fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'parameters': sum(p.numel() for p in model.parameters()),
              'study_permutation_max_abs_error': permutation_error,
              'masked_nan_max_abs_error': missing_error,
              'real_feature_forward_backward_finite': True, 'folds': fold_counts,
              'training_started': False, 'accuracy_measured': False,
              'cuda_execution_verified': False}
    (run / 'classifier_head_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
