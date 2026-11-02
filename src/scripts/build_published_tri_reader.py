"""Stage a private inference package for the user-linked recipe; never launch."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public_kernels/published_tri_reader'
RUN = ROOT / 'runs/published_tri_reader'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    RUN.mkdir(exist_ok=True)
    assert not (RUN / 'launch.json').exists(), 'Observe any existing launch before editing'
    research = ROOT / 'research/tri_reader_20261010'
    result = json.loads((ROOT / 'runs/tri_reader_20261010/result.json').read_text())
    protocol = json.loads((ROOT / 'runs/tri_reader_20261010/protocol.json').read_text())
    notebook = next(research.glob('*.ipynb'))
    assert sha(notebook) == result['source_sha256']
    assert result['status'] == 'EXACT_COMPONENT_REPLAY_COMPLETE_FULL_NOTEBOOK_NOT_EXECUTED'
    tree = ast.parse((ROOT / 'public_kernels/fixed_reader_blend_v3/rsna-knee-n02-fixed-reader-blend-v3.py').read_text())
    modules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MODULES' for t in n.targets))
    source_notebook = ROOT / 'research/new_public/nartaa_0945_20261009/rsna-knee-0945-efficient-224crop.ipynb'
    crop = ''.join(json.loads(source_notebook.read_text())['cells'][4]['source'])
    assert hashlib.sha256(crop.encode()).hexdigest() == 'e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c'
    spec = importlib.util.spec_from_file_location('n03_builder', ROOT / 'scripts/build_n03_paired_diagnostic.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Reuse previously exercised hash/load/no-view/failed-worker guards.
    checkpoint = '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
    crop = helper.checked_224(crop, checkpoint).replace('/kaggle/working/n03_raw.csv', '/kaggle/working/tri_reader_crop_raw.csv')
    modules['crop224.py'] = crop
    modules['published_tri_reader.py'] = (ROOT / 'inference/published_tri_reader.py').read_text()
    modules['CREDITS-TRI-READER.txt'] = ('Public tri-reader recipe: rabari9999; CoAtNet checkpoints/inference: nartaa; '
        'Raptor architecture: dreaddevelopment; ConvNeXt reader: goodpjw2008; '
        'baseline fixed recipe: sujanmajhisuzan. Preserve original CREDITS and README-Nartaa. '
        'The claimed0.954 private priors are not released or included. No new raw external data.\n')
    contract = {
        'schema': 'frozen_public_tri_reader_v1',
        'source_url': 'https://www.kaggle.com/code/rabari9999/rsna-knee-sota-tri-reader-pipeline-lb-0-954',
        'source_sha256': sha(notebook), 'crop_weight': protocol['recipe']['crop_weight'],
        'reader_weights': protocol['recipe']['reader_weights'],
        'crop_checkpoint_sha256': checkpoint,
        'visible_seconds_cap': 1200, 'worker_seconds_cap': 7200,
        'scoring_allocation_hours': 2.2, 'free_gpu_required_hours': 3.2,
        'source_diagnostic_result_sha256': sha(ROOT / 'runs/tri_reader_20261010/result.json'),
        'model_or_weight_search': False, 'competition_submission_enabled': False,
        'limitations': ['Advertised0.954 private priors and proprietary blend are not reproduced',
                       'Public recipe showed a positive reused-cohort diagnostic with interval including zero',
                       'Full hidden execution and official score remain unverified'],
        'module_sha256': {name: hashlib.sha256(code.encode()).hexdigest() for name, code in modules.items()},
    }
    modules['tri_reader_contract.json'] = json.dumps(contract, indent=2) + '\n'
    for name, code in modules.items():
        if name.endswith('.py'):
            compile(code, name, 'exec')
    code = 'import runpy,sys\nfrom pathlib import Path\nMODULES=' + repr(modules) + '''
root=Path('/kaggle/working/tri_reader_source');root.mkdir(exist_ok=False)
for name,source in MODULES.items():
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
sys.path.insert(0,str(root))
runpy.run_path(str(root/'published_tri_reader.py'),run_name='__main__')
'''
    slug = 'rsna-knee-n04-published-tri-reader'
    filename = slug + '.py'
    compile(code, filename, 'exec')
    (OUT / filename).write_text(code)
    metadata = json.loads((ROOT / 'public_kernels/fixed_reader_blend_v3/kernel-metadata.json').read_text())
    metadata.update(id='adityakumarrai0401/' + slug, title='RSNA Knee N04 Published Tri Reader', code_file=filename)
    (OUT / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (RUN / 'contract.json').write_text(modules['tri_reader_contract.json'])
    build = {'status': 'STAGED_NOT_LAUNCHED', 'kernel': metadata['id'],
             'code_sha256': sha(OUT / filename), 'metadata_sha256': sha(OUT / 'kernel-metadata.json'),
             'contract_sha256': sha(RUN / 'contract.json'), 'builder_sha256': sha(Path(__file__)),
             'diagnostic_result_sha256': contract['source_diagnostic_result_sha256'],
             'next_gate': 'Evaluate existing E05fold4 paired output, then fresh quota/job/input checks before GPU execution',
             'full_notebook_execution_complete': False, 'competition_submission_enabled': False}
    (RUN / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
    print(json.dumps(build, indent=2))


if __name__ == '__main__':
    main()
