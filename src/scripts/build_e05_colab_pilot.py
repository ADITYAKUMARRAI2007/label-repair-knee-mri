"""Package unchanged E04 definitions plus a paired one-epoch resource test."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'runs/e05_colab_pilot';out.mkdir(exist_ok=True)
    original=ROOT/'public_kernels/e04_convnext_folds_01/e04_full_fold_worker_preview.py'
    source=original.read_text();nodes=ast.parse(source).body;pieces=[]
    for node in nodes:
        if isinstance(node,ast.FunctionDef) and node.name=='main':continue
        if isinstance(node,ast.If) and '__name__' in ast.unparse(node.test):continue
        pieces.append(ast.get_source_segment(source,node))
    definitions='\n\n'.join(pieces).replace('/kaggle/','/content/kaggle/')+'\n'
    transform=ROOT/'training/e05_confidence.py';pilot=ROOT/'training/e05_resource_epoch.py'
    code=definitions+'\n'+transform.read_text()+'\n'+pilot.read_text()
    compile(code,'worker.py','exec')
    (out/'worker.py').write_text(code)
    protocol=ROOT/'runs/e05_confidence_ablation/protocol.json'
    build={'status':'READY_FOR_RESOURCE_EXECUTION','code_sha256':sha(out/'worker.py'),
           'sources':{str(p.relative_to(ROOT)):sha(p) for p in [original,transform,pilot,protocol]},
           'control_seed_equals_candidate_seed':True,'fold':0,'epochs_per_arm':1,
           'arms':['control','uniform'],'outer_runtime_cap_seconds':2700,
           'total_session_budget_seconds':3600,'accuracy_measured':False,
           'competition_submission_enabled':False}
    (out/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps({'worker_bytes':len(code),'compressed_bytes':len(zlib.compress(code.encode(),9)),**build}))


if __name__=='__main__':main()
