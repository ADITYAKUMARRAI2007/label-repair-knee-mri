"""Stage an input-only ablation;leave the qualified public candidate unchanged."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/window_integrity_20261010'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    package = ROOT / 'public_kernels/published_tri_reader/rsna-knee-n04-published-tri-reader.py'
    tree = ast.parse(package.read_text())
    modules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(x, ast.Name) and x.id == 'MODULES' for x in n.targets))
    original = modules['anchor/published_inference.py']
    old = '    return [cs[i] for i in idx]'
    new = '    return coherent_centers([cs[i] for i in idx], mask, SLOT_SIZES)'
    assert original.count(old) == 1
    modified = 'from coherent_triplets import coherent_centers\n' + original.replace(old, new)
    old_shape = '    assert plain.shape==mirrored.shape==(94,3,320,320)'
    new_shape = '    assert plain.shape == mirrored.shape and plain.ndim == 4 and plain.shape[0] > 0 and plain.shape[1:] == (3,320,320)'
    assert modified.count(old_shape) == 1
    modified = modified.replace(old_shape, new_shape)
    compile(modified, 'anchor_coherent.py', 'exec')
    helper = (ROOT / 'inference/coherent_triplets.py').read_text()
    protocol = {'schema': 'single_coherent_anchor_input_ablation_v1',
                'hypothesis': 'Exclude triplets spanning acquisitions or missing slots from the384-trained anchor',
                'control': 'Exact current Nartaa320-crop anatomical-mirror inference',
                'change': 'Filter original selected centers;allow dynamic nonempty window count',
                'preserved': ['Checkpoint bytes', 'Resolution', 'Physical crop and normalization',
                              'Anatomical mirror probability average', 'Other model branches', 'Fixed published blend weights'],
                'control_source_sha256': digest(original), 'candidate_source_sha256': digest(modified),
                'helper_sha256': digest(helper),
                'evaluation': 'Reexecute both arms on the same fixed78-study diagnostic cohort;58expert comparison plus whole-pipeline fixed-blend comparison',
                'promotion_rule': 'Input correctness alone is insufficient;review paired accuracy,per-target regressions and uncertainty before promotion',
                'prior_evidence': 'Public linear tri-reader diagnostic results known;this variant has no prediction or score',
                'gpu_gate': 'Evaluate existing E05 job and the already prepared public tri-reader candidate before another GPU experiment',
                'weight_or_resolution_search': False, 'competition_submission_enabled': False,
                'limitations': ['Original training may exploit the removed windows',
                               'Actual decoded missing masks may differ from metadata predictions',
                               'Expert cohort reused;paired results would remain descriptive']}
    OUT.mkdir(exist_ok=True)
    (OUT / 'anchor_published.py').write_text(original)
    (OUT / 'anchor_coherent.py').write_text(modified)
    (OUT / 'coherent_triplets.py').write_text(helper)
    (OUT / 'ablation_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'status': 'COHERENT_ANCHOR_SOURCE_STAGED_NOT_RUN',
                      'candidate_source_sha256': digest(modified), 'gpu_launched': False,
                      'current_public_candidate_unchanged': True}))


if __name__ == '__main__':
    main()
