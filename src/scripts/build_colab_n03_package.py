"""Adapt device/version checks only; preserve the frozen 78-study comparison."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/colab_n03'


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    original=(ROOT/'public_kernels/n03_paired_diagnostic/diagnostic.py').read_text()
    tree=ast.parse(original)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body[:4]}
    modules=values['MODULES'];before={n:sha(v) for n,v in modules.items()}
    source=modules['fixed_reader_blend.py']
    assert source.count('assert torch.cuda.device_count() == 2')==1
    source=source.replace('assert torch.cuda.device_count() == 2','assert torch.cuda.device_count() == 1')
    modules['fixed_reader_blend.py']=source
    for name,source in list(modules.items()):
        if name.endswith('.py'):
            modules[name]=source.replace('/kaggle/', '/content/kaggle/')
    anchor_contract=json.loads(modules['anchor/contract.json'])
    anchor_contract['source_sha256']=sha(modules['anchor/published_inference.py'])
    modules['anchor/contract.json']=json.dumps(anchor_contract,indent=2)
    contract=json.loads(modules['contract.json'])
    contract['torch_version']='2.11.0+cu130'
    contract['module_sha256']={n:sha(v) for n,v in modules.items() if n!='contract.json'}
    modules['contract.json']=json.dumps(contract,indent=2)
    values['COHORT']=values['COHORT'].replace('/kaggle/', '/content/kaggle/')
    code='\n'.join(name+'='+repr(values[name]) for name in ['MODULES','STUDIES','COHORT','READER_WEIGHTS'])+'\n'
    code+='\n'.join(original.splitlines()[4:]).replace('/kaggle/', '/content/kaggle/')+'\n'
    compile(code,'colab_diagnostic.py','exec')
    for name,source in modules.items():
        if name.endswith('.py'):compile(source,name,'exec')
    after={n:sha(v) for n,v in modules.items()}
    changed=[n for n in before if before[n]!=after[n]]
    assert 'fixed_reader_blend.py' in changed and 'contract.json' in changed
    (OUT/'diagnostic.py').write_text(code)
    protocol=json.loads((ROOT/'runs/n03_paired_diagnostic/protocol.json').read_text())
    build={'status':'COLAB_STAGED','source_sha256':sha(original),'code_sha256':sha(code),
           'changed_modules':changed,'changes':['One A100 instead of two T4 devices','Torch build 2.11.0+cu130 instead of 2.11.0+cu128','Storage paths moved from /kaggle to writable /content/kaggle'],
           'unchanged':['Cohort','Checkpoints','Preprocessing','Sampling','TTA','Blend weights','Reader parameters'],
           'parity_limitation':'GPU architecture and CUDA build may change floating-point results; paired comparison uses the same runtime for both recipes.',
           'screenshot_compute_units':52.4,'screenshot_units_per_hour':5.3,
           'total_session_budget_hours':1,'runtime_cap_seconds':1200,
           'protocol_sha256':sha((ROOT/'runs/n03_paired_diagnostic/protocol.json').read_text()),
           'module_sha256':after,'competition_submission_enabled':False}
    (OUT/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    files={'diagnostic.py':code,'build.json':json.dumps(build),
           'decoder_identity.json':(ROOT/'models/convnext_decoder_cp313/decoder_identity.json').read_text()}
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode()
    cell='''import base64,zlib,json,sys,subprocess
from pathlib import Path
payload=PAYLOAD
files=json.loads(zlib.decompress(base64.b64decode(payload)))
setup=Path('/content/rsna_setup')
for name,content in files.items(): (setup/name).write_text(content)
decoder=Path('/content/kaggle/input/rsna-knee-decoder-cp313');decoder.mkdir(parents=True,exist_ok=True)
(decoder/'decoder_identity.json').write_text(files['decoder_identity.json'])
download=subprocess.run([sys.executable,'-m','pip','download','-q','--no-deps','--dest',str(decoder),'pylibjpeg==2.1.0','pylibjpeg-libjpeg==2.4.0','pylibjpeg-openjpeg==2.6.0'],capture_output=True,text=True,timeout=180)
assert download.returncode==0, 'Pinned decoder wheel staging failed'
Path('/content/kaggle/working').mkdir(parents=True,exist_ok=True)
print(json.dumps({'status':'COLAB_CODE_STAGED','code_sha256':files['build.json'] and json.loads(files['build.json'])['code_sha256'],'model_job_started':False}))
'''.replace('PAYLOAD',repr(payload))
    request={'action':'call','name':'add_code_cell','arguments':{'cellIndex':4,'language':'python','code':cell}}
    (OUT/'stage_request.json').write_text(json.dumps(request)+'\n')
    print(json.dumps({'code_bytes':len(code),'transfer_request_bytes':(OUT/'stage_request.json').stat().st_size,'changed_modules':changed,'gpu_job_started':False}))


if __name__=='__main__':main()
