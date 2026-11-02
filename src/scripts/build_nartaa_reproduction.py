"""Stage one private reproduction of the exact published 0.949 model; no push."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    local = ROOT / 'runs/nartaa_reproduction_v1'
    assert not (local / 'build.json').exists() and not (local / 'launch.json').exists()
    research = ROOT / 'research/new_public/refresh_20261008'
    source = research / 'nartaa_mirror/scored_source.py'
    guard = ROOT / 'inference/nartaa_reproduction_guard.py'
    manifest = json.loads((research / 'nartaa_weights/checkpoint-manifest.json').read_text())['accuracy']
    assert sha(source) == manifest['source_sha256']
    checks = json.loads((local / 'guard_check.json').read_text())
    assert checks['status'] == 'PASS_EXECUTION_AND_OUTPUT_GUARDS' and checks['source_sha256'] == sha(guard)
    model_dir = ROOT / 'models/nartaa_publication_20261007'
    identity = json.loads((model_dir / 'identity.json').read_text())
    assert identity['sha256'] == manifest['sha256'] and identity['status'] == 'PASS_PUBLIC_CHECKPOINT_IDENTITY'
    numeric = json.loads((model_dir / 'checkpoint_inspection.json').read_text())
    assert numeric['status'] == 'PASS_SAFE_LOAD_AND_FINITE_TENSORS'
    contract = dict(schema='nartaa_exact_public_reproduction_v1',
        source_url='https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror',
        source_sha256=manifest['source_sha256'], checkpoint_filename=manifest['filename'],
        checkpoint_sha256=manifest['sha256'], author_reported_public_score='0.949',
        author_reported_submission=manifest['submission'], our_public_score=None,
        torch_version='2.11.0+cu128', timm_version='1.0.29', worker_seconds_cap=7200,
        qualification_allocation_hours=2.1, reserve_gpu_hours=2.0,
        prediction_recipe='Original source bytes; single384-trained CoAtNet;96 slices;94 windows;320 center crop;equal plain/anatomical-mirror probability average',
        selection='Fixed published standalone model; no blend or parameter search',
        prerequisite='Evaluate accepted OrthoDiffusion v2 output before any new GPU launch',
        competition_submission_enabled=False,
        limitations=['Gold58 reused in published model development; not independent validation',
                     'Reported public score belongs to the author until reproduced on our account',
                     'Wrapper rejects reported study failures; original per-slice recovery stays unchanged'])
    contract_path = local / 'contract.json'
    contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    modules = {'published_inference.py': source.read_text(),
               'nartaa_reproduction_guard.py': guard.read_text(),
               'contract.json': contract_path.read_text(),
               'README-upstream.md': (research / 'nartaa_weights/README.md').read_text()}
    for name, code in modules.items():
        if name.endswith('.py'):
            compile(code, name, 'exec')
    script = """import runpy
from pathlib import Path
MODULES = """ + repr(modules) + """
root=Path('/kaggle/working/nartaa_source');root.mkdir(exist_ok=False)
for name,content in MODULES.items():
    (root/name).write_text(content)
runpy.run_path(str(root/'nartaa_reproduction_guard.py'),run_name='__main__')
"""
    directory = ROOT / 'public_kernels/nartaa_reproduction_v1'
    directory.mkdir(exist_ok=True)
    slug = 'rsna-knee-n01-nartaa-reproduction'
    code_path = directory / (slug + '.py')
    compile(script, code_path.name, 'exec')
    code_path.write_text(script)
    metadata = json.loads((research / 'nartaa_mirror/kernel-metadata.json').read_text())
    metadata.pop('id_no', None)
    metadata.update(id='adityakumarrai0401/' + slug, title='RSNA Knee N01 Published Model Reproduction',
                    code_file=code_path.name, kernel_type='script', is_private=True)
    meta_path = directory / 'kernel-metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    sources = [source, guard, Path(__file__), code_path, meta_path, contract_path,
               research / 'nartaa_weights/README.md']
    gates = [local / 'guard_check.json', model_dir / 'identity.json', model_dir / 'checkpoint_inspection.json',
             research / 'nartaa_weights/checkpoint-manifest.json']
    build = dict(status='STAGED_NOT_LAUNCHED', contract=contract,
                 sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                 gates={str(p.relative_to(ROOT)): sha(p) for p in gates},
                 new_gpu_jobs=0, new_competition_submissions=0)
    (local / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
    print(json.dumps(dict(status=build['status'], kernel=metadata['id'], bytes=code_path.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
