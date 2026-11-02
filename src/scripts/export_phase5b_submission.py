"""Export the tested inference with only a stricter hidden-run time budget."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=read(RUN/'serving_audit.json');assert audit['status']=='PASS_VISIBLE_COMBINED_INFERENCE_READY_FOR_REVIEW'
    original=ROOT/'public_kernels/phase5b_fixed_compact_blend/inference.ipynb'
    build=read(RUN/'serving_build.json');assert sha(original)==build['notebook_sha256']
    notebook=read(original);cell=next(c for c in notebook['cells'] if c['cell_type']=='code')
    tree=ast.parse(''.join(cell['source']))
    modules=ast.literal_eval(tree.body[0].value);old=read(RUN/'serving_contract.json')
    assert json.loads(modules['serving_contract.json'])==old
    new=dict(old);new['hidden_seconds_cap']=31800
    assert new['hidden_seconds_cap']<32400
    modules['serving_contract.json']=json.dumps(new,indent=2)+'\n'
    code='MODULES='+repr(modules)+'\nexec(compile(MODULES["phase5b_serving_driver.py"],"phase5b_serving_driver.py","exec"))\nrun(MODULES)\n'
    ast.parse(code);cell['source']=code.splitlines(keepends=True)
    package=ROOT/'public_kernels/phase5b_submission_candidate';package.mkdir(exist_ok=True)
    target=package/'inference.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n')
    meta=read(ROOT/'public_kernels/phase5b_fixed_compact_blend/kernel-metadata.json')
    (package/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    assert {k:(old[k],new[k]) for k in old if old[k]!=new[k]}=={'hidden_seconds_cap':(34200,31800)}
    result={'status':'KAGGLE_OFFLINE_SUBMISSION_EXPORT_READY_FOR_REVIEW','notebook':str(target.relative_to(ROOT)),
            'notebook_sha256':sha(target),'code_cell_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'qualified_native_version':audit['version'],'qualified_kernel':audit['kernel'],
            'prediction_code_and_weights_identical_to_executed_package':True,
            'only_difference':{'hidden_seconds_cap':{'executed_visible_package_value':34200,'export_value':31800}},
            'visible_seconds_cap_unchanged':True,'export_native_execution':False,'hidden_native_execution':False,
            'platform_commit_timeout_seconds':1800,'official_competition_runtime_max_seconds':32400,
            'additional_GPU_job_launched':False,'competition_submission_enabled':False,
            'next_step_after_explicit_approval':'Publish this concrete export as the next version with a bounded visible commit, then request official notebook evaluation. Verify final accepted source hashes and artifacts before submission.'}
    (RUN/'submission_export.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
