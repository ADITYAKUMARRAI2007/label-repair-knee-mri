"""Build an offline, frozen A--E inference package. Does not launch or submit."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
PACKAGE=ROOT/'public_kernels/phase5b_fixed_compact_blend'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    assert not (RUN/'serving_launch.json').exists()
    baseline=ROOT/'public_kernels/fixed_reader_blend_v3/rsna-knee-n02-fixed-reader-blend-v3.py'
    assert sha(baseline)=='ca4dbebfc379e45064d6cd7441c281d4f05114d54f426d68c0a4a34e653f2032'
    tree=ast.parse(baseline.read_text())
    payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODULES' for t in n.targets))
    n02=read(ROOT/'runs/fixed_reader_blend_v3/contract.json')
    assert json.loads(payload['contract.json'])==n02
    for n,h in n02['module_sha256'].items():assert hashlib.sha256(payload[n].encode()).hexdigest()==h,n
    protocol=read(ROOT/'research/public_refresh_20261011/qualification_protocol.json')
    modules={'n02/'+n:s for n,s in payload.items()}
    exact=read(RUN/'latest/source_sha.json')
    for n,h in exact.items():
        p=RUN/'latest/modules'/n;assert sha(p)==h,n
        modules['compact/'+n]=p.read_text()
    modules['raptor_source.py']=(RUN/'latest/raptor_source.py').read_text()
    modules['phase5b_serving_driver.py']=(ROOT/'inference/phase5b_serving_driver.py').read_text()
    modules['reader_entry.py']='import sys\nfrom pathlib import Path\nroot=Path(sys.argv[1])\nsys.path.insert(0,str(root))\nfrom fixed_reader_blend import reader_main\nreader_main(root)\n'
    for n,s in modules.items():
        if n.endswith('.py'):ast.parse(s,filename=n)
    reference=pd.read_csv(ROOT/'runs/fixed_reader_blend_v3/outputs/submission.csv',dtype={'StudyInstanceUID':str})
    assert len(reference)==3
    contract={'schema':'phase5b_offline_fixed_candidates_v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in modules.items()},
              'code_sha256':protocol['code_sha256'],'manifest_sha256':protocol['manifest_sha256'],
              'decoder_sha256':n02['decoder_wheels']['cp313'],'target_order':protocol['target_order'],
              'torch_version':protocol['torch_version'],'visible_seconds_cap':1500,'hidden_seconds_cap':34200,
              'platform_timeout_seconds':1800,'allocated_gpu_count':2,'reserve_gpu_hours':.25,
              'visible_N02_reference':reference.to_dict('records'),'primary_candidate':'C_N02_compact',
              'primary_weights':{'N02':.7,'compact':.3},'weight_search':False,
              'competition_submission_enabled':False,'full_hidden_runtime_measured':False,
              'source_note':'Exact retrieved Jenil V4 model and pixel functions, hardened failure checks and performance chunk64/one loader. V3 unavailable.',
              'qualification_receipt_sha256':sha(ROOT/'research/public_refresh_20261011/qualification_output/helios_qualification_attempt_tkewxzd0/execution_receipt.json'),
              'baseline_package_sha256':sha(baseline)}
    modules['serving_contract.json']=json.dumps(contract,indent=2)+'\n'
    code='MODULES='+repr(modules)+'\nexec(compile(MODULES["phase5b_serving_driver.py"],"phase5b_serving_driver.py","exec"))\nrun(MODULES)\n'
    ast.parse(code)
    notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# RSNA Knee: fixed N02 + compact reader\n','Offline inference. Preserves candidates A–E; primary C uses fixed 70/30 percentile ranks. No submission API. Displayed author V3 score has not been reproduced on this account.']},
                       {'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'outputs':[],'execution_count':None}],
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    meta=read(ROOT/'public_kernels/helios_compact_qualification/kernel-metadata.json')
    meta.update(id='adityakumarrai0401/rsna-knee-phase5b-fixed-compact-blend',title='RSNA Knee Phase5B Fixed Compact Blend',code_file='inference.ipynb',
                dataset_sources=read(ROOT/'public_kernels/fixed_reader_blend_v3/kernel-metadata.json')['dataset_sources']+[
                    'heliosli/rsna-knee-compact-v1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004'])
    PACKAGE.mkdir(exist_ok=True)
    (PACKAGE/'inference.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    (PACKAGE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (RUN/'serving_contract.json').write_text(modules['serving_contract.json'])
    build={'status':'BUILT_NOT_LAUNCHED','kernel':meta['id'],'notebook_sha256':sha(PACKAGE/'inference.ipynb'),
           'metadata_sha256':sha(PACKAGE/'kernel-metadata.json'),'code_cell_sha256':hashlib.sha256(code.encode()).hexdigest(),
           'protocol_sha256':sha(RUN/'serving_contract.json'),'baseline_unchanged_sha256':sha(baseline)}
    (RUN/'serving_build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps(build))
if __name__=='__main__':main()
