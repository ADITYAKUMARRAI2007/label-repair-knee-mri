"""Freeze one complementary classifier before any MARS task predictions exist."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = RUN / 'classifier_protocol.json'
    if path.exists():
        print(json.dumps({'status': 'EXISTING_PROTOCOL_PRESERVED', 'sha256': sha(path)}))
        return
    frozen_path = ROOT / 'runs/E04_contract/e04_contract.json'
    frozen = json.loads(frozen_path.read_text())
    assert len(frozen['rows']) == 4407
    assert sum(r['expert'] for r in frozen['rows']) == 58
    arrays = {}
    for name, expected in frozen['arrays'].items():
        p = frozen_path.parent / name
        assert sha(p) == expected
        arrays[str(p.relative_to(ROOT))] = expected
    baseline = ROOT / 'runs/colab_n03/output/n02_diagnostic.csv'
    evidence = ROOT / 'runs/window_integrity_20261010/pair_evaluation.json'
    protocol = {
        'schema': 'mars_frozen_spatial_classifier_v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'rationale': 'Test complementary full-volume three-plane MRI features after negative uniform-weight and acquisition-filter experiments.',
        'checkpoint_sha256': '88388a59d92935a15d30bd41c0e9dbece99ba78db73cb374d44290fd6566ccc3',
        'encoder_trainable': False, 'features_shape': [3, 768, 3, 3, 3],
        'planes': ['Axial', 'Coronal', 'Sagittal'],
        'head': {'name': 'MarsSpatialHead', 'hidden': 256, 'attention_heads': 8,
                 'dropout': .1, 'target_queries': 12,
                 'position': 'relative canonical superior/anterior/right cube coordinates'},
        'target_order': frozen['target_order'], 'epochs': 30, 'batch_studies': 64,
        'optimizer': 'AdamW', 'learning_rate': .0003, 'weight_decay': .01,
        'gradient_clip': 3., 'seed': 20261010, 'dtype': 'float32',
        'checkpoint_rule': 'final_epoch_30_only_no_expert_checkpoint_selection',
        'independent_heads_per_fold': True, 'folds': [0, 1, 2, 3, 4],
        'heldout_report_groups_excluded': True,
        'targets_and_weights': frozen['fusion'],
        'sources_sha256': {str(frozen_path.relative_to(ROOT)): sha(frozen_path), **arrays,
                           'training/mars_spatial_head.py': sha(ROOT / 'training/mars_spatial_head.py'),
                           'training/mars_classifier_train.py': sha(ROOT / 'training/mars_classifier_train.py'),
                           'data/train.csv': frozen['official_train_sha256']},
        'baseline_path': str(baseline.relative_to(ROOT)), 'baseline_sha256': sha(baseline),
        'fixed_rank_blend': {'baseline': .9, 'candidate': .1, 'target_weights_searched': False},
        'evaluation': {'expert_studies': 58, 'requires_complete_five_fold_oof': True,
                       'bootstrap_groups': 'report_group', 'bootstrap_draws': 2000,
                       'bootstrap_seed': 20261010,
                       'promotion_review': 'Fixed blend delta above0.001 and paired95%lower bound above0; then inspect serving parity before submitting'},
        'prior_window_evidence_sha256': sha(evidence),
        'execution_dependency': 'Measured MARS pilot numerical parity, cost and full4407study feature coverage must pass before fitting',
        'automatic_gpu_launch': False, 'automatic_competition_submission': False,
        'expected_public_score': None,
        'limitations': ['Gold58 was repeatedly reused and is a rejection screen, not independent confirmation.',
                       'Public report-label tuning independence is unverified.',
                       'Report groups do not establish patient/site separation.',
                       'Author knee preprocessing is unavailable; the input adapter is our explicit RSNA adaptation.',
                       'One held-out head per expert differs from a five-head serving ensemble.'],
    }
    with path.open('x') as handle:
        json.dump(protocol, handle, indent=2); handle.write('\n')
    print(json.dumps({'status': 'CLASSIFIER_RECIPE_FROZEN_BEFORE_TASK_PREDICTIONS',
                      'protocol_sha256': sha(path), 'training_started': False}))


if __name__ == '__main__':
    main()
