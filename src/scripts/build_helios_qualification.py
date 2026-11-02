"""Package one fixed, supervised public-reader qualification; no upload or launch."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/public_refresh_20261011'
PACKAGE=ROOT/'public_kernels/helios_compact_qualification'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (RUN/'qualification_launch.json').exists()
    module_dir=RUN/'helios_restored/modules'
    hashes=json.loads((RUN/'helios_restored/source_sha.json').read_text())
    for n,h in hashes.items(): assert sha(module_dir/n)==h,n
    code_hashes=json.loads((RUN/'helios_restored/code_sha.json').read_text())
    for n,h in code_hashes.items(): assert sha(RUN/'foundation_docs'/n)==h,n
    for n in ['anatomy.py','anatomy_runtime.py','geometry_bank.py','data.py']:
        code_hashes[n]=sha(RUN/'foundation_docs'/n)
    manifest=json.loads((RUN/'compact_docs/manifest.json').read_text())
    split=json.loads((ROOT/'runs/E03_contract/split.json').read_text())
    experts=sorted(split['gold']); assert len(experts)==58
    old=json.loads((ROOT/'research/foundation_refresh_20261010/MARS/resource_protocol.json').read_text())
    resources=list(old['studies'][:2])
    for reason in ['maximum_maximum_native_extent_mm','maximum_maximum_obliquity',
                   'maximum_maximum_native_slice_count','maximum_maximum_native_pixels']:
        ids=[u for u,s in old['selection'].items() if reason in s['reasons']]
        assert len(ids)==1,reason
        if ids[0] not in resources: resources.append(ids[0])
    for uid in old['studies']:
        if len(resources)>=6: break
        if uid not in resources: resources.append(uid)
    assert len(resources)==6 and not set(resources)&set(experts)
    modules={'author/'+n:(module_dir/n).read_text() for n in hashes}
    modules['worker_entry.py']='''import sys,json,time\nfrom pathlib import Path\nimport torch\nimport compact_infer\nif __name__=="__main__":\n    torch.set_num_threads(2)\n    torch.cuda.reset_peak_memory_stats()\n    try:\n        compact_infer.main()\n    finally:\n        out=Path(sys.argv[sys.argv.index("--output")+1]);out.mkdir(parents=True,exist_ok=True)\n        (out/"gpu_memory.json").write_text(json.dumps({"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}))\n'''
    modules['helios_qualification_driver.py']=(ROOT/'inference/helios_qualification_driver.py').read_text()
    contract={'schema':'helios_compact_native_qualification_v1',
              'recorded_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in modules.items()},
              'code_sha256':code_hashes,'manifest_sha256':sha(RUN/'compact_docs/manifest.json'),
              'target_order':json.loads((ROOT/'runs/E04_contract/e04_contract.json').read_text())['target_order'],
              'resource_ids':resources,'expert_ids':experts,'torch_version':'2.11.0+cu128',
              'maximum_parent_seconds':2400,'platform_timeout_seconds':2700,
              'reserve_gpu_hours':.25,'allocated_gpu_count':2,
              'prediction_replay_max_absolute_error':.001,
              'predictor':'Four strict-loaded ConvNeXt-small folds; mean logits then sigmoid',
              'model_parameters':{k:manifest[k] for k in ['backbone','size','crop','views','slices','phys_mm','eval_stride']},
              'inference_changes':'Performance settings only: chunk64 primary, chunk32 replay, one loader per GPU. Exact author pixel/model source.',
              'primary_blend':{'baseline':'N02 scored0.950 reader','baseline_weight':.7,'compact_weight':.3,
                               'rank_method':'average percentile per target','weight_search':False},
              'primary_blend_is_exact_published_recipe':False,
              'published_recipe_difference':'Published70/30 uses Raptor alone; this fixed adaptation uses our already-scored N02 baseline.',
              'baseline_path':'runs/colab_n03/output/n02_diagnostic.csv',
              'baseline_sha256':sha(ROOT/'runs/colab_n03/output/n02_diagnostic.csv'),
              'diagnostic_gate':'Complete zero-fallback native prediction chain, replay<=0.001, blenddelta>0.001 and pairedreportgroup95pctlower>0. Passing warrants serving review only.',
              'independent_clinical_validation':False,
              'limitations':['Gold58 has been repeatedly reused by us; author inspected some geometry settings on Gold58.',
                             'Public author training exclusions are claims, not independently verified.',
                             'Small runtime sample does not prove complete hidden test coverage.',
                             'Bootstrap intervals omit model-selection and leaderboard uncertainty.'],
              'competition_submission_enabled':False,'label_values_embedded':False,
              'test_time_fitting':False,'images_or_features_embedded':False}
    modules['qualification_contract.json']=json.dumps(contract,indent=2)+'\n'
    code='MODULES='+repr(modules)+'\nexec(compile(MODULES["helios_qualification_driver.py"],"helios_qualification_driver.py","exec"))\nrun(MODULES)\n'
    ast.parse(code)
    notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Restored compact reader: native qualification\n','Bounded decoding, resource and replay test followed by a fixed expert diagnostic. No competition submission.']},
                       {'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'outputs':[],'execution_count':None}],
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    meta=json.loads((ROOT/'public_kernels/mars_frozen_classifier/kernel-metadata.json').read_text())
    meta.update(id='adityakumarrai0401/rsna-knee-helios-compact-qualification',
                title='RSNA Knee Helios Compact Qualification',code_file='qualification.ipynb',
                dataset_sources=['heliosli/rsna-knee-compact-v1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
                                 'adityakumarrai0401/rsna-knee-decoder-cp313'],kernel_sources=[])
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE/'qualification.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    (PACKAGE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (RUN/'qualification_protocol.json').write_text(modules['qualification_contract.json'])
    result={'status':'BUILT_NOT_LAUNCHED','kernel':meta['id'],
            'notebook_sha256':sha(PACKAGE/'qualification.ipynb'),
            'metadata_sha256':sha(PACKAGE/'kernel-metadata.json'),
            'code_cell_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'protocol_sha256':sha(RUN/'qualification_protocol.json'),'source_sha256':contract['source_sha256']}
    (RUN/'qualification_build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))

if __name__=='__main__':main()
