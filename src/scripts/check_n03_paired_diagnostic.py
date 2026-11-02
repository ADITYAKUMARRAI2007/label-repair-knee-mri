"""Exercise changed failure paths without CUDA or model inference."""
import ast
import hashlib
import json
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'public_kernels/n03_paired_diagnostic/diagnostic.py'
    tree = ast.parse(path.read_text())
    assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body[:4]}
    modules = assignments['MODULES']
    contract = json.loads(modules['contract.json'])
    for name, expected in contract['module_sha256'].items():
        assert hashlib.sha256(modules[name].encode()).hexdigest() == expected, name
    candidate = ast.parse(modules['candidate.py'])
    functions = {n.name: n for n in candidate.body if isinstance(n, ast.FunctionDef)}
    def extract(node, scope):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<guard-test>', 'exec'), scope)
        return scope[node.name]
    scope = {'np':np, 'MAXS':3, 'IMG':2, '_SER':{}, '_TSDIR':'unused'}
    pre = extract(functions['_pre_one'], scope)
    vol, mask = np.ones((3,2,2),np.uint8), np.ones(3,np.uint8)
    scope['_READER'] = lambda *args: (vol, mask)
    sid, actual, actual_mask = pre('study')
    assert sid == 'study' and actual is vol and actual_mask is mask
    scope['_READER'] = lambda *args: (vol, np.zeros(3,np.uint8))
    try:
        pre('empty')
        raise AssertionError('Empty study accepted')
    except RuntimeError as error:
        assert 'No decoded view' in str(error)
    def broken(*args):
        raise ValueError('Injected decode failure')
    scope['_READER'] = broken
    try:
        pre('broken')
        raise AssertionError('Broken study accepted')
    except RuntimeError as error:
        assert 'No decoded view' in str(error)
    # Real file hashing must reject mismatched bytes before deserialization.
    calls = []
    scope = {'torch':SimpleNamespace(load=lambda *a,**kw:calls.append(kw))}
    load_model = extract(functions['load_model'], scope)
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp)/'checkpoint.pt'; checkpoint.write_bytes(b'wrong model')
        try:
            load_model(checkpoint,'arch',224,'cpu')
            raise AssertionError('Wrong checkpoint accepted')
        except RuntimeError as error:
            assert 'checkpoint bytes differ' in str(error)
    assert not calls
    # A failed consumer must record the failure, complete queue bookkeeping,
    # and reach the final rejection gate rather than silently keeping 0.5.
    main_node = functions['main']
    consumer_node = next(n for n in main_node.body if isinstance(n,ast.FunctionDef) and n.name=='consumer')
    q = queue.Queue(); q.put(('study',vol,mask)); q.put(None)
    scope = {'q':q,'pos':{'study':0},'models':{'cpu':[(object(),224)]},'xla':None,
             'gpu_windows':lambda *args:None,'infer_probs':broken,'K_EVAL':2,
             'arm_probs':[np.full((1,12),.5)],'failures':[],'lock':threading.Lock(),
             'done_n':[0],'N':2}
    extract(consumer_node,scope)('cpu')
    assert scope['failures'] and q.unfinished_tasks == 0 and scope['done_n'] == [1]
    gate = next(n for n in main_node.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='failures')
    try:
        exec(compile(ast.Module(body=[gate],type_ignores=[]),'<gate-test>','exec'),scope)
        raise AssertionError('Consumer failure accepted')
    except RuntimeError as error:
        assert 'inference failed' in str(error)
    for name in ['candidate.py','anchor/published_inference.py']:
        fn = next(n for n in ast.parse(modules[name]).body if isinstance(n,ast.FunctionDef) and n.name=='find_test_root')
        assert isinstance(fn.body[0],ast.Return) and fn.body[0].value.value == assignments['COHORT']
    studies=assignments['STUDIES']; assert len(studies)==len(set(studies))==78
    protocol=json.loads((ROOT/'runs/n03_paired_diagnostic/protocol.json').read_text())
    assert set(protocol['diagnostic_ids']) <= set(studies) and not protocol['weight_search_allowed']
    report={'status':'PASS_LOCAL_CHANGED_PATHS','code_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'checks':['embedded module identities','real checkpoint hash mismatch rejects before torch.load',
                      'valid decoded arrays preserved','empty and failed study rejected',
                      'consumer failure recorded and rejected without queue deadlock',
                      'both sources route to the same 78-study cohort','fixed weights; no search'],
            'not_tested':['actual checkpoint safe loading','CUDA inference','Kaggle dependencies','prediction accuracy'],
            'remote_launch':False}
    (ROOT/'runs/n03_paired_diagnostic/local_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
