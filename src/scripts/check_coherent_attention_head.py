"""Check the published pooling head supports filtered window counts on CPU."""
import ast
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/window_integrity_20261010'


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261010)
    sources = {}
    classes = []
    for arm in ['anchor_published.py', 'anchor_coherent.py']:
        path = RUN / arm
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'RaptorClassifier')
        classes.append(ast.dump(node, include_attributes=False))
        sources[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert classes[0] == classes[1], 'The ablation changed the pooling architecture'
    namespace = {'nn': torch.nn, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<published_head>', 'exec'), namespace)
    # Identity operates on synthetic per-window features: no image backbone is
    # executed and no pretrained checkpoint or task accuracy is tested here.
    model = namespace['RaptorClassifier'](torch.nn.Identity(), F_dim=32).cpu().eval()
    checks = []
    with torch.inference_mode():
        for windows in [94, 86, 66, 1]:
            features = torch.randn(1, windows, 32)
            logits = model(features)
            assert logits.shape == (1, 12) and torch.isfinite(logits).all()
            order = torch.randperm(windows)
            permuted = model(features[:, order])
            repeated = model(features.repeat(1, 2, 1))
            errors = {'permutation_max_abs': float((permuted-logits).abs().max()),
                      'uniform_duplication_max_abs': float((repeated-logits).abs().max())}
            assert max(errors.values()) < 1e-6
            checks.append({'windows': windows, 'logit_shape': list(logits.shape), **errors})
    result = {'status': 'PASS_CPU_DYNAMIC_WINDOW_POOLING_HEAD',
              'identical_control_candidate_model_definition': True,
              'source_sha256': sources, 'checks': checks,
              'backbone': 'Identity on synthetic32dimensional features; no images encoded',
              'gpu_used': False, 'checkpoint_loaded': False, 'task_accuracy_measured': False,
              'limitations': ['Does not verify CUDA/fp16 or pretrained image encoder execution.',
                             'Pooling support does not establish any accuracy benefit from filtering.']}
    (RUN / 'attention_head_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
