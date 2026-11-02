"""Freeze a single published V3 blend; no upload or scoring."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'runs/fixed_reader_blend_v2';out.mkdir(exist_ok=True)
    assert not (out/'build.json').exists() and not (out/'launch.json').exists()
    research=ROOT/'research/new_public/after_n01_20261008/fixed_v3_components'
    identity=json.loads((research/'identity.json').read_text())
    check=json.loads((out/'guard_check.json').read_text())
    runtime=ROOT/'inference/fixed_reader_blend_v2.py'
    assert check['status']=='PASS_BLEND_AND_FAILURE_GUARDS' and check['runtime_sha256']==sha(runtime)
    prior=ROOT/'runs/submission_monitor_n01_20261008/result.json'
    result=json.loads(prior.read_text())['submissions'][0]
    assert result['status']=='COMPLETE' and result['submission_id']==56961667 and result['public_score']=='0.949'
    failure=ROOT/'runs/fixed_reader_blend_v1/failure_audit.json'
    assert json.loads(failure.read_text())['status']=='CONFIRMED_PACKAGING_FAILURE'
    model=ROOT/'models/convnext_public_reader'
    assert json.loads((model/'inspection.json').read_text())['status']=='PASS_SAFE_LOAD_FINITE'
    for name,r in json.loads((model/'identity.json').read_text())['files'].items():assert sha(model/name)==r['sha256']
    source=ROOT/'research/new_public/refresh_20261008/nartaa_mirror/scored_source.py'
    guard=ROOT/'inference/nartaa_reproduction_guard.py'
    modules={'fixed_reader_blend.py':runtime.read_text(),'nartaa_reproduction_guard.py':guard.read_text(),
             'anchor/nartaa_reproduction_guard.py':guard.read_text(),'anchor/published_inference.py':source.read_text(),
             'anchor/contract.json':(ROOT/'runs/nartaa_reproduction_v1/contract.json').read_text(),
             'README-Nartaa.md':(ROOT/'research/new_public/refresh_20261008/nartaa_weights/README.md').read_text()}
    for name,r in identity['reader_components'].items():
        assert sha(research/name)==r['sha256'];modules['reader/'+name]=(research/name).read_text()
    modules['CREDITS.txt']='Nartaa: 0.949 checkpoint and inference; dreaddevelopment: Raptor architecture; GoodPJW2008: ConvNeXt reader (Apache-2.0); sujanmajhisuzan: published fixed ApexV3 recipe. Preserve README-Nartaa.md including OAI acknowledgement. No raw external data used.\n'
    contract=dict(schema='fixed_published_reader_blend_v2',torch_version='2.11.0+cu128',timm_version='1.0.29',
        repair='Same-version cp313 decoder wheels; verify decoder imports before model execution',
        author_reported_score='0.950',our_score=None,anchor_official_score='0.949',anchor_submission=56961667,
        source_url='https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-apex-grandmaster-stack',
        weights=identity['fixed_convnext_weights'],reader_windows=16,reader_same_plane_slot_fill=True,
        reader_checkpoints=json.loads((model/'identity.json').read_text())['files'],
        decoder_wheels=json.loads((ROOT/'models/convnext_decoder_cp313/decoder_identity.json').read_text()),
        visible_seconds_cap=600,worker_seconds_cap=7200,qualification_allocation_hours=0.5,
        scoring_allocation_hours=2.1,reserve_gpu_hours=1.0,competition_submission_enabled=False,
        module_sha256={name:hashlib.sha256(code.encode()).hexdigest() for name,code in modules.items()},
        limitations=['Author score is not our score; no score guarantee.',
            'Gold58 reused; no independent paired validation.',
            'Reader differs from B03 in slice count and missing-slot handling.',
            'Qualification checks visible execution only; no leaderboard weight search.'])
    modules['contract.json']=json.dumps(contract,indent=2)+'\n'
    for name,code in modules.items():
        if name.endswith('.py'):compile(code,name,'exec')
    script='import runpy,sys\nfrom pathlib import Path\nMODULES='+repr(modules)+'''\nroot=Path('/kaggle/working/fixed_blend_source');root.mkdir(exist_ok=False)
for name,code in MODULES.items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(code)
sys.path.insert(0,str(root))
runpy.run_path(str(root/'fixed_reader_blend.py'),run_name='__main__')
'''
    directory=ROOT/'public_kernels/fixed_reader_blend_v2';directory.mkdir(exist_ok=True)
    slug='rsna-knee-n02-fixed-reader-blend-v2';code=directory/(slug+'.py');code.write_text(script);compile(script,code.name,'exec')
    metadata=json.loads((ROOT/'public_kernels/nartaa_reproduction_v1/kernel-metadata.json').read_text())
    metadata.update(id='adityakumarrai0401/'+slug,title='RSNA Knee N02 Fixed Reader Blend V2',code_file=code.name,
                    dataset_sources=['nartaa/rsna-knee-publication-swa-weights-20261007','goodpjw2008/rsna-knee-2-5d-convnext-reader','adityakumarrai0401/rsna-knee-decoder-cp313'])
    meta=directory/'kernel-metadata.json';meta.write_text(json.dumps(metadata,indent=2)+'\n')
    cp=out/'contract.json';cp.write_text(modules['contract.json'])
    sources=[runtime,guard,source,Path(__file__),code,meta,cp]+list(research.glob('*.py'))
    sources += [ROOT/'scripts'/name for name in ['launch_fixed_reader_blend_v2.py',
                 'audit_fixed_reader_blend_v2.py','watch_fixed_reader_blend_v2.py','check_fixed_reader_blend_v2.py']]
    gates=[out/'guard_check.json',model/'identity.json',model/'inspection.json',ROOT/'models/convnext_decoder_cp313/decoder_identity.json',prior,research/'identity.json']
    gates += [failure,ROOT/'runs/fixed_reader_blend_v1/decoder_upload.json',ROOT/'models/convnext_decoder_cp313/wheel_audit.json']
    build=dict(status='STAGED_NOT_LAUNCHED',contract=contract,sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
               gates={str(p.relative_to(ROOT)):sha(p) for p in gates})
    (out/'build.json').write_text(json.dumps(build,indent=2)+'\n');print(json.dumps({'status':build['status'],'kernel':metadata['id'],'bytes':code.stat().st_size}))


if __name__=='__main__':main()
