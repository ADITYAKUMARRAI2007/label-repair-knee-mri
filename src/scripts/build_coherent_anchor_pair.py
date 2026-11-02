"""Package the frozen input ablation without uploading or allocating a GPU."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/window_integrity_20261010'
PACKAGE = ROOT / 'public_kernels/coherent_anchor_pair'
COHORT = '/kaggle/working/coherent_pair_cohort'


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:80]
    return source.replace(old, new)


def instrument(source):
    # Apply identical observation and routing edits to the two sources.
    source = 'import hashlib,json\n' + source
    source = replace_once(source, 'def find_test_root():',
                          f'def find_test_root():\n    return {COHORT!r}')
    source = replace_once(source, '    return sid, vol, mask',
                          '    if not np.any(mask):\n        raise RuntimeError("No decoded study")\n    return sid, vol, mask')
    before = '            try:\n                for a, (model, res) in enumerate(models[d]):'
    after = '''            try:
                centers = _eval_centers(mask, len(mask), K_EVAL)
                slot = np.repeat(np.arange(len(SLOT_SIZES)), SLOT_SIZES)
                coherent = [c for c in centers if mask[c-1:c+2].all() and (slot[c-1:c+2] == slot[c]).all()]
                print("PAIRED_INPUT_AUDIT " + json.dumps({"study": sid,
                    "volume_sha256": hashlib.sha256(vol.tobytes()).hexdigest(),
                    "mask": mask.tolist(), "shape": list(vol.shape),
                    "windows": len(centers), "coherent_windows": len(coherent)}), flush=True)
                for a, (model, res) in enumerate(models[d]):'''
    return replace_once(source, before, after)


def main():
    assert not (RUN / 'pair_launch.json').exists(), 'Do not overwrite an accepted experiment'
    protocol_path = RUN / 'ablation_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    control = (RUN / 'anchor_published.py').read_text()
    candidate = (RUN / 'anchor_coherent.py').read_text()
    helper = (RUN / 'coherent_triplets.py').read_text()
    assert sha(control) == protocol['control_source_sha256']
    assert sha(candidate) == protocol['candidate_source_sha256']
    assert sha(helper) == protocol['helper_sha256']
    n04 = ROOT / 'public_kernels/published_tri_reader/rsna-knee-n04-published-tri-reader.py'
    tree = ast.parse(n04.read_text())
    original = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'MODULES' for target in node.targets))
    assert original['anchor/published_inference.py'] == control
    anchor = json.loads(original['anchor/contract.json'])
    baseline = json.loads(original['contract.json'])
    cohort = json.loads((ROOT / 'runs/n03_paired_diagnostic/protocol.json').read_text())
    studies = cohort['cohort']
    assert len(studies) == len(set(studies)) == 78 and len(cohort['diagnostic_ids']) == 58
    decoder = (ROOT / 'runs/published_tri_reader/output/tri_reader_source/check_decoders.py').read_text()
    assert "'/kaggle/working/reader_decoder_env'" in decoder
    modules = {'control.py': instrument(control), 'candidate.py': instrument(candidate),
               'coherent_triplets.py': helper, 'check_decoders.py': decoder,
               'nartaa_reproduction_guard.py': original['nartaa_reproduction_guard.py'],
               'coherent_anchor_pair.py': (ROOT / 'inference/coherent_anchor_pair.py').read_text()}
    contract = {'schema': 'coherent_anchor_pair_execution_v1', 'studies': studies,
                'diagnostic_ids': cohort['diagnostic_ids'], 'cohort_path': COHORT,
                'protocol_sha256': sha(protocol_path.read_text()),
                'checkpoint_filename': anchor['checkpoint_filename'],
                'checkpoint_sha256': anchor['checkpoint_sha256'],
                'torch_version': anchor['torch_version'], 'timm_version': anchor['timm_version'],
                'decoder_wheels': baseline['decoder_wheels']['cp313'],
                'runtime_cap_seconds': 1800, 'gpu_allocation_hours': 0.65,
                'module_sha256': {name: sha(source) for name, source in modules.items()},
                'accuracy_measured': False, 'competition_submission_enabled': False,
                'gpu_gate': protocol['gpu_gate'], 'weight_search_allowed': False,
                'comparison': 'Both anchor arms execute on the same78 studies; decode hashes and masks must match. Locally replay each with the identical archived reader and224-crop predictions on this exact cohort.',
                'limitations': protocol['limitations']}
    modules['pair_contract.json'] = json.dumps(contract, indent=2) + '\n'
    for name, source in modules.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    driver = '''
from pathlib import Path
import sys
root=Path('/kaggle/working/coherent_pair_source');root.mkdir(exist_ok=False)
for name,source in MODULES.items():
    (root/name).write_text(source)
sys.path.insert(0,str(root))
from coherent_anchor_pair import main
main()
'''
    code = 'MODULES=' + repr(modules) + '\n' + driver
    compile(code, 'pair.py', 'exec')
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE / 'pair.py').write_text(code)
    meta = json.loads((n04.parent / 'kernel-metadata.json').read_text())
    meta.update(id='adityakumarrai0401/rsna-knee-coherent-anchor-pair',
                title='RSNA Knee Coherent Anchor Pair', code_file='pair.py')
    meta['dataset_sources'] = [source for source in meta['dataset_sources']
                               if source != 'goodpjw2008/rsna-knee-2-5d-convnext-reader']
    (PACKAGE / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    report = {'status': 'PAIRED_INPUT_ABLATION_PACKAGED_NOT_LAUNCHED', 'code_sha256': sha(code),
              'metadata_sha256': sha((PACKAGE / 'kernel-metadata.json').read_text()),
              'contract_sha256': sha(modules['pair_contract.json']),
              'n04_code_sha256': sha(n04.read_text()), 'new_gpu_jobs': 0,
              'new_competition_submissions': 0, 'models_executed': False}
    (RUN / 'pair_build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
