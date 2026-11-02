"""Stage one fixed 224-crop comparison against the scored N02 recipe. No upload."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public_kernels/n03_paired_diagnostic"
RUN = ROOT / "runs/n03_paired_diagnostic"
COHORT = "/kaggle/working/n03_cohort"


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:120]
    return source.replace(old, new)


def route(source):
    # Prefer a private training-cohort facade; the rest of the reader is unchanged.
    start = source.index('    cands = [', source.index('def find_test_root():'))
    source = source[:start] + f'    return {COHORT!r}\n' + source[start:]
    return source


def checked_224(source, checkpoint_sha):
    old = '    ck = torch.load(pt_path, map_location="cpu", weights_only=False)'
    new = f'''    import hashlib
    with open(pt_path, "rb") as handle:
        observed_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if observed_sha != {checkpoint_sha!r}:
        raise RuntimeError("224 checkpoint bytes differ")
    print("N03_CHECKPOINT_SHA256 " + observed_sha, flush=True)
    ck = torch.load(pt_path, map_location="cpu", weights_only=True)
    if ck.get("arch", arch_default) != arch_default or int(ck.get("res", res_default)) != 224:
        raise RuntimeError("224 checkpoint architecture/resolution differs")
    if not all(torch.isfinite(t).all() for t in ck["model"].values()):
        raise RuntimeError("224 checkpoint contains nonfinite tensors")'''
    source = replace_once(source, old, new)
    source = replace_once(source, '    return sid, vol, mask', '''    if not np.any(mask):
        raise RuntimeError("No decoded view for study " + sid)
    return sid, vol, mask''')
    source = replace_once(source, '    models = {}', '    failures = []\n    models = {}')
    source = replace_once(source, '            with lock:\n                done_n[0] += 1',
                          '                failures.append((sid, repr(e)))\n            with lock:\n                done_n[0] += 1')
    source = replace_once(source, '    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).', '''    if failures:
        raise RuntimeError("224 inference failed for studies: " + repr(failures))
    if done_n[0] != N or not all(np.isfinite(p).all() for p in arm_probs):
        raise RuntimeError("224 incomplete or nonfinite predictions")
    raw = pd.DataFrame(arm_probs[0], columns=LAB)
    raw.insert(0, "StudyInstanceUID", test_ids)
    raw.to_csv("/kaggle/working/n03_raw.csv", index=False)
    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).''')
    return source


def main():
    OUT.mkdir(exist_ok=True)
    RUN.mkdir(exist_ok=True)
    assert not (RUN / 'launch.json').exists(), 'Already launched: do not replace frozen source'
    original = ROOT / 'public_kernels/fixed_reader_blend_v3/rsna-knee-n02-fixed-reader-blend-v3.py'
    tree = ast.parse(original.read_text())
    modules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MODULES' for t in n.targets))
    original_hashes = {name: sha(code) for name, code in modules.items()}
    source_path = ROOT / 'research/new_public/nartaa_0945_20261009/rsna-knee-0945-efficient-224crop.ipynb'
    source = ''.join(json.loads(source_path.read_text())['cells'][4]['source'])
    assert sha(source) == 'e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c'
    checkpoint_sha = '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
    modules['candidate.py'] = route(checked_224(source, checkpoint_sha))
    modules['anchor/published_inference.py'] = route(modules['anchor/published_inference.py'])
    # Reject an entirely undecodable study; preserve individual acquisition recovery.
    modules['anchor/published_inference.py'] = replace_once(
        modules['anchor/published_inference.py'], '    return sid, vol, mask',
        '    if not np.any(mask):\n        raise RuntimeError("No decoded view for study " + sid)\n    return sid, vol, mask')
    ac = json.loads(modules['anchor/contract.json'])
    ac['source_sha256'] = sha(modules['anchor/published_inference.py'])
    modules['anchor/contract.json'] = json.dumps(ac, indent=2)
    runtime = modules['fixed_reader_blend.py']
    old = "        data = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),\n                     Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]\n                    if (p / 'test.csv').is_file())"
    modules['fixed_reader_blend.py'] = replace_once(runtime, old, f'        data = Path({COHORT!r})')
    contract = json.loads(modules['contract.json'])
    contract['worker_seconds_cap'] = 900
    contract['module_sha256'] = {name: sha(code) for name, code in modules.items() if name != 'contract.json'}
    modules['contract.json'] = json.dumps(contract, indent=2)
    split = json.loads((ROOT / 'runs/E03_contract/split.json').read_text())
    weak = json.loads((ROOT / 'runs/fixed_reader_blend_v3/qualification_cohort.json').read_text())['studies']
    gold = sorted(split['gold'])
    assert len(gold) == 58 and len(weak) == 20 and not set(gold) & set(weak)
    studies = weak + gold
    protocol = {
        'status': 'STAGED_NOT_LAUNCHED', 'cohort': studies, 'diagnostic_ids': gold,
        'candidate_checkpoint_sha256': checkpoint_sha,
        'recipe': '70% N01 rank + 30% 224 rank, then the unchanged N02 per-target reader weights',
        'candidate_anchor_weight': 0.30, 'reader_weights': contract['weights'],
        'weight_search_allowed': False, 'competition_submission_enabled': False,
        'gold_limit': 'Repeatedly inspected and used in public model development; descriptive rejection screen only. Positive results do not establish independent improvement.',
        'runtime_cap_seconds': 1200, 'gpu_allocation_hours': 0.5,
        'free_gpu_required_before_launch_hours': 3.0,
        'launch_policy': 'Recheck live quota, pending jobs and input availability. Never launch using paid capacity. No automatic submission.',
        'next_gate': 'Replay predictions and compare the single frozen blend with N02; inspect failure counts, per-class results and paired uncertainty before any submission decision.',
        'original_module_sha256': original_hashes,
    }
    (RUN / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # Ground-truth values are intentionally absent from the runnable package.
    driver = '''import csv,json,os,runpy,sys,time
from pathlib import Path
root=Path('/kaggle/working/n03_diagnostic_source');root.mkdir(exist_ok=False)
for name,code in MODULES.items():
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code)
sys.path.insert(0,str(root))
from nartaa_reproduction_guard import run_source,validate_submission,sha
data=next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')] if (p/'train.csv').is_file())
cohort=Path(COHORT);cohort.mkdir(exist_ok=False)
with (data/'train.csv').open() as f:
    available={r['StudyInstanceUID'] for r in csv.DictReader(f)}
assert set(STUDIES)<=available
with (cohort/'test.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['StudyInstanceUID']);w.writerows([[u] for u in STUDIES])
with (data/'sample_submission.csv').open() as f: columns=next(csv.reader(f))
with (cohort/'sample_submission.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(columns);w.writerows([[u]+[0.5]*12 for u in STUDIES])
with (data/'train_series.csv').open() as f:
    reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if r['StudyInstanceUID'] in set(STUDIES)]
with (cohort/'test_series.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
series=next(p for p in [data/'train_series',data/'train_images'] if p.is_dir())
(cohort/'test_series').symlink_to(series,target_is_directory=True)
work=Path('/kaggle/working');start=time.monotonic()
try:
    baseline=run_source(root/'fixed_reader_blend.py',work/'n02_diagnostic.log',900,work)
    (work/'submission.csv').replace(work/'n02_diagnostic.csv')
    remaining=1200-(time.monotonic()-start)
    if remaining<=0: raise TimeoutError('Diagnostic runtime exhausted')
    candidate=run_source(root/'candidate.py',work/'n03_diagnostic.log',remaining,work)
    validate_submission(work/'submission.csv',cohort/'test.csv',cohort/'sample_submission.csv')
    (work/'submission.csv').replace(work/'n03_diagnostic.csv')
    from fixed_reader_blend import blend
    weights={label:0.30 for label in columns[1:]}
    blend(work/'fixed_reader_blend/anchor.csv',work/'n03_diagnostic.csv',work/'spatial_anchor.csv',weights,cohort/'test.csv',cohort/'sample_submission.csv')
    blend(work/'spatial_anchor.csv',work/'fixed_reader_blend/reader.csv',work/'candidate_diagnostic.csv',READER_WEIGHTS,cohort/'test.csv',cohort/'sample_submission.csv')
    report={'status':'PREDICTIONS_READY_NOT_SCORED','rows':len(STUDIES),'baseline':baseline,'candidate':candidate,'seconds':time.monotonic()-start,'competition_submission_enabled':False,
            'files':{n:sha(work/n) for n in ['n02_diagnostic.csv','n03_diagnostic.csv','candidate_diagnostic.csv','n03_raw.csv']}}
    (work/'diagnostic_receipt.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    if (work/'submission.csv').exists(): (work/'submission.csv').replace(work/'rejected_diagnostic.csv')
'''
    code = 'MODULES=' + repr(modules) + '\nSTUDIES=' + repr(studies) + '\nCOHORT=' + repr(COHORT)
    code += '\nREADER_WEIGHTS=' + repr(contract['weights']) + '\n' + driver
    for name, value in modules.items():
        if name.endswith('.py'): compile(value, name, 'exec')
    compile(code, 'diagnostic.py', 'exec')
    (OUT / 'diagnostic.py').write_text(code)
    meta = json.loads((ROOT / 'public_kernels/fixed_reader_blend_v3/kernel-metadata.json').read_text())
    meta.update(id='adityakumarrai0401/rsna-knee-n03-paired-diagnostic', title='RSNA Knee N03 Paired Diagnostic', code_file='diagnostic.py')
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    build = {'status':'STAGED_NOT_LAUNCHED','code_sha256':sha(code),'metadata_sha256':sha((OUT/'kernel-metadata.json').read_text()),'protocol_sha256':sha((RUN/'protocol.json').read_text()),'models_executed':False,'submission_created':False}
    build['launcher_sha256'] = sha((ROOT/'scripts/launch_n03_paired_diagnostic.py').read_text())
    (RUN / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
    print(json.dumps(build))


if __name__ == '__main__':
    main()
