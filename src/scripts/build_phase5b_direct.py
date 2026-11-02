"""Package the fixed published V4 recipe, skipping N02's additional reader."""
import ast,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
OUT=RUN/'direct_recipe';PACKAGE=ROOT/'public_kernels/phase5b_published_direct'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'launch.json').exists()
 original=ROOT/'public_kernels/phase5b_submission_candidate/inference.ipynb'
 assert sha(original)==read(RUN/'submission_export.json')['notebook_sha256']
 book=read(original);code=''.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
 old=ast.literal_eval(ast.parse(code).body[0].value);c=json.loads(old['serving_contract.json'])
 driver=old['phase5b_serving_driver.py']
 a=driver.index("        assets=locate('goodpjw2008'");b=driver.index("        bundle=locate('heliosli'",a)
 driver=driver[:a]+driver[b:]
 a=driver.index("        candidates={'B_published_V4'");b=driver.index('        receipt={',a)
 driver=driver[:a]+'''        f=pd.DataFrame(b.astype(np.float32),columns=labels);f.insert(0,'StudyInstanceUID',ids)
        f.to_csv(attempt/'B_published_V4.csv',index=False)
        result=checked(attempt/'B_published_V4.csv')
        parity=None
        if len(ids)==3:
            expected=pd.DataFrame(c['visible_B_reference']).set_index('StudyInstanceUID').loc[ids,labels]
            parity=float(abs(result.to_numpy()-expected.to_numpy()).max())
            assert parity<=1e-6,f'Direct published recipe differs from qualified B:{parity}'
'''+driver[b:]
 driver=driver.replace("str(overlay)+os.pathsep+str(source/'n02')","str(overlay)")
 driver=driver.replace("'old_reader':old,",'').replace("'raptor':raptor,'old_reader':old,'compact':cp","'raptor':raptor,'compact':cp")
 driver=driver.replace('visible_N02_parity_max_abs_error','visible_B_parity_max_abs_error').replace('C_N02_compact.csv','B_published_V4.csv').replace('COMPLETE_OFFLINE_A_TO_E_COMBINED_INFERENCE','COMPLETE_OFFLINE_PUBLISHED_V4_DIRECT_INFERENCE')
 driver=driver.replace('phase5b_attempt_','phase5b_direct_attempt_').replace('phase5b_latest_attempt.json','phase5b_direct_latest_attempt.json').replace('phase5b.lock','phase5b_direct.lock')
 ast.parse(driver);assert 'old_reader' not in driver and 'fixed_reader_blend' not in driver
 modules={k:v for k,v in old.items() if k.startswith('compact/') or k=='raptor_source.py'}
 modules['direct_driver.py']=driver
 modules['README-Nartaa.md']=old['n02/README-Nartaa.md']
 ref=RUN/'approved_output'/read(RUN/'approved_output/phase5b_latest_attempt.json')['attempt_directory']/'B_published_V4.csv'
 for k in ['visible_N02_reference','primary_weights','baseline_package_sha256']:c.pop(k,None)
 c.update(schema='phase5b_published_V4_direct_v1',recorded_utc=datetime.now(timezone.utc).isoformat(),
          source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in modules.items()},
          visible_B_reference=pd.read_csv(ref,dtype={'StudyInstanceUID':str}).to_dict('records'),
          primary_candidate='B_published_V4',primary_weights={'raptor':.7,'compact':.3},
          visible_seconds_cap=1200,hidden_seconds_cap=31800,platform_timeout_seconds=1500,
          source_note='Published V4 fixed recipe. No original N02 ConvNeXt reader. Same verified Raptor/compact pixel and model code, fixed70/30 percentile ranks, zero study fallbacks required.',
          unchanged_reference_sha256=sha(ref))
 modules['serving_contract.json']=json.dumps(c,indent=2)+'\n'
 code='MODULES='+repr(modules)+'\nexec(compile(MODULES["direct_driver.py"],"direct_driver.py","exec"))\nrun(MODULES)\n';ast.parse(code)
 book['cells']=[{'cell_type':'markdown','metadata':{},'source':['# Published V4 direct ensemble\n','CoAtNet + four compact ConvNeXt-small folds; fixed70/30 percentile ranks. One controlled reference, no coefficient search. Exact historical V3 source remains unavailable.']},{'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'outputs':[],'execution_count':None}]
 meta=read(ROOT/'public_kernels/phase5b_submission_candidate/kernel-metadata.json')
 meta.update(id='adityakumarrai0401/rsna-knee-phase5b-published-direct',title='RSNA Knee Phase5B Published Direct')
 meta['dataset_sources']=[x for x in meta['dataset_sources'] if not x.startswith('goodpjw2008/')]
 PACKAGE.mkdir(exist_ok=True);(PACKAGE/'inference.ipynb').write_text(json.dumps(book,indent=1)+'\n');(PACKAGE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 (OUT/'contract.json').write_text(modules['serving_contract.json'])
 record={'status':'BUILT_NOT_LAUNCHED','kernel':meta['id'],'notebook_sha256':sha(PACKAGE/'inference.ipynb'),'metadata_sha256':sha(PACKAGE/'kernel-metadata.json'),'code_cell_sha256':hashlib.sha256(code.encode()).hexdigest(),'contract_sha256':sha(OUT/'contract.json'),'test_gate':'Source hashes, zero study fallbacks, exact B reference parity, finite complete12target predictions, fixed recipe and runtime reliability.','reason_for_one_official_evaluation':'Measure the published baseline and the value/runtime cost of the extra old reader; no blend-weight search.','our_current_best_public_score':.954,'our_current_best_submission':57070500,'new_training':False,'new_submission':False}
 (OUT/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
