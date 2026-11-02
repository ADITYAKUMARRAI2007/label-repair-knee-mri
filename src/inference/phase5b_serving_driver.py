"""Offline fixed A--E inference, one execution per unique model family, no submission API."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def locate(owner,slug,sentinel):
    a=[p for p in [Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug] if (p/sentinel).exists()]
    assert len(a)==1,(slug,a)
    return a[0]

def run(modules):
    work=Path('/kaggle/working');lock=(work/'phase5b.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=Path(tempfile.mkdtemp(prefix='phase5b_attempt_',dir=work))
    write(work/'phase5b_latest_attempt.json',{'attempt_directory':attempt.name})
    source=attempt/'source';source.mkdir();started=time.monotonic();children=[];published=False
    try:
        assert not (work/'submission.csv').exists()
        for n,s in modules.items():
            p=Path(n);assert not p.is_absolute() and '..' not in p.parts
            q=source/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
        c=json.loads(modules['serving_contract.json'])
        for n,h in c['source_sha256'].items():assert sha(source/n)==h,n
        import torch,timm
        assert str(torch.__version__)==c['torch_version'] and timm.__version__=='1.0.29'
        assert sys.version_info[:2]==(3,13) and torch.cuda.device_count()==2
        data=next(p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'test.csv').exists())
        import pandas as pd
        import numpy as np
        ids=pd.read_csv(data/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
        assert len(ids)==len(set(ids)) and ids
        labels=c['target_order'];columns=['StudyInstanceUID',*labels]
        assert list(pd.read_csv(data/'sample_submission.csv',nrows=0).columns)==columns
        cap=c['visible_seconds_cap'] if len(ids)<=3 else c['hidden_seconds_cap']
        mem=[]
        def remaining():
            n=cap-(time.monotonic()-started)-30
            assert n>0,'Combined serving cap exhausted'
            return n
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        def execute(name,cmd,stage_cap,custom_env=None):
            log=(attempt/(name+'.log')).open('w');start=time.monotonic();last=0
            p=subprocess.Popen(cmd,cwd=work,env=custom_env or env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
            try:
                while p.poll() is None:
                    if time.monotonic()-start>stage_cap or time.monotonic()-started>cap-20:raise TimeoutError(name+' exceeded fixed cap')
                    if time.monotonic()-last>20:
                        r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                        mem.append({'stage':name,'seconds':time.monotonic()-started,'gpus':r.stdout.strip()});write(attempt/'resource_samples.json',mem)
                        last=time.monotonic()
                    time.sleep(2)
                assert p.returncode==0,f'{name} exited{p.returncode}; see{attempt/name}.log'
                return {'status':'COMPLETE','seconds':time.monotonic()-start,'log_sha256':sha(attempt/(name+'.log'))}
            finally:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    p.wait(timeout=10)
                log.close()
        def checked(path):
            f=pd.read_csv(path,dtype={'StudyInstanceUID':str})
            assert list(f.columns)==columns and f.StudyInstanceUID.is_unique and set(f.StudyInstanceUID)==set(ids),str(path)
            f=f.set_index('StudyInstanceUID').loc[ids,labels]
            a=f.to_numpy();assert np.isfinite(a).all() and ((0<=a)&(a<=1)).all()
            return f
        decoder=locate('adityakumarrai0401','rsna-knee-decoder-cp313','decoder_identity.json')
        for n,h in c['decoder_sha256'].items():assert sha(decoder/n)==h,n
        overlay=attempt/'decoder_env'
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(overlay),*[str(decoder/n) for n in c['decoder_sha256']]],check=True,timeout=120)
        env['PYTHONPATH']=str(overlay)+os.pathsep+str(source/'n02')
        raptor=execute('raptor',[sys.executable,'-u',str(source/'raptor_source.py')],min(7200,remaining()))
        log=(attempt/'raptor.log').read_text(errors='replace')
        assert not re.search(r'\[pre\].*\bFAILED\b|\[gpu[^\]]*\].*\bFALLBACK\b',log)
        assert f'[gpu] done: {len(ids)}/{len(ids)}' in log
        raw=checked(work/'raptor_raw.csv');anchor=checked(work/'raptor_sub.csv')
        shutil.copyfile(work/'raptor_raw.csv',attempt/'raptor_raw.csv')
        shutil.copyfile(work/'raptor_sub.csv',attempt/'raptor_sub.csv')
        assets=locate('goodpjw2008','rsna-knee-2-5d-convnext-reader','cnxt_v0_fold0.pt')
        n02_contract=json.loads(modules['n02/contract.json'])
        for n,v in n02_contract['reader_checkpoints'].items():assert sha(assets/n)==v['sha256'],n
        runtime={'data':str(data),'checkpoints':[str(assets/n) for n in sorted(n02_contract['reader_checkpoints'])],
                 'output':str(attempt/'old_reader.csv'),'qualification':False}
        write(source/'n02/reader_runtime.json',runtime)
        old=execute('old_reader',[sys.executable,'-u',str(source/'reader_entry.py'),str(source/'n02')],min(7200,remaining()))
        checked(attempt/'old_reader.csv')
        sys.path.insert(0,str(source/'n02'))
        from fixed_reader_blend import blend
        blend(work/'raptor_sub.csv',attempt/'old_reader.csv',attempt/'A_N02.csv',n02_contract['weights'],data/'test.csv',data/'sample_submission.csv')
        base=checked(attempt/'A_N02.csv')
        bundle=locate('heliosli','rsna-knee-compact-v1','manifest.json')
        code_root=locate('heliosli','rsna-orthofoundation-fold0-epoch7-20261004','receipt.json')
        manifest=json.loads((bundle/'manifest.json').read_text())
        assert sha(bundle/'manifest.json')==c['manifest_sha256']
        for n,h in manifest['sha256'].items():assert sha(bundle/n)==h,n
        code=code_root/'cloud_code'
        for n,h in c['code_sha256'].items():assert sha(code/'training'/n)==h,n
        assert sha(code_root/'anatomy.pt')==manifest['anchor_weights_sha256']
        wheels=sorted((code_root/'dependencies/wheels').glob('*.whl'))
        assert {p.name for p in wheels}=={'pydicom-3.0.1-py3-none-any.whl','timm-1.0.25-py3-none-any.whl'}
        compact_env=attempt/'compact_env'
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(compact_env),*map(str,wheels)],check=True,timeout=min(120,remaining()))
        ce=dict(env,PYTHONPATH=os.pathsep.join(map(str,[compact_env,overlay,code,source/'compact'])))
        pred=attempt/'compact_predictions';budget=remaining()
        args=['--backbone',manifest['backbone'],'--size',str(manifest['size']),'--crop',str(manifest['crop']),
              '--eval-stride',str(manifest['eval_stride']),'--views',manifest['views'],
              '--slices',str(manifest['slices']),'--phys-mm',str(manifest['phys_mm']),
              '--state-spec',str(bundle/manifest['state_spec']),'--side-model',str(bundle/manifest['side_model']),
              '--side-threshold',str(manifest['side_threshold']),'--side-position-mm',str(manifest['side_position_mm']),
              '--side-max-x',str(manifest['side_max_x']),'--anchor-weights',str(code_root/'anatomy.pt'),'--chunk','64']
        cp=execute('compact',[sys.executable,'-u',str(source/'compact/probe_dual.py'),'--data',str(data),'--split','test',
                               '--code',str(code),'--bundle',str(bundle),'--arm',manifest['arm'],'--output',str(pred),
                               '--amp','auto','--loader-workers','1','--deadline',str(time.time()+budget-10),
                               '--worker','compact_infer.py','--worker-args',json.dumps(args),'--max-fallback-fraction','0'],budget,ce)
        rc=json.loads((pred/'submission_receipt.json').read_text())
        assert rc['fallback_studies']==0 and rc['scored']==len(ids)
        assert rc['status']=='PASS' and rc['rows']==len(ids)
        assert all(x==0 for x in rc['worker_exit_codes'].values()),rc['worker_exit_codes']
        seen=[]
        for wd in sorted(pred.glob('worker[0-9]*')):
            if not (wd/'predictions.csv').exists():continue
            wr=json.loads((wd/'receipt.json').read_text())
            assert wr['status']=='PASS' and not wr['fallback'] and not wr['timed_out']
            assert wr['models']==manifest['models'] and wr['scored']==wr['studies']
            assert wr['anchor_weights_sha256']==manifest['anchor_weights_sha256']
            assert wr['side_check']['model_sha256']==manifest['sha256'][manifest['side_model']]
            for n,h in wr['sha256'].items():assert sha(source/'compact'/n)==h,n
            f=pd.read_csv(wd/'predictions.csv',dtype={'StudyInstanceUID':str})
            assert f.StudyInstanceUID.is_unique;seen+=f.StudyInstanceUID.tolist()
        assert len(seen)==len(set(seen))==len(ids) and set(seen)==set(ids)
        compact=checked(pred/'submission.csv')
        compact.reset_index().to_csv(attempt/'compact_raw.csv',index=False)
        def ranks(f):return f.rank(method='average',pct=True).to_numpy()
        b=.7*ranks(raw)+.3*ranks(compact)
        candidates={'B_published_V4':b,'C_N02_compact':.7*ranks(base)+.3*ranks(compact),
                    'D_N02_published':.7*ranks(base)+.3*pd.DataFrame(b).rank(method='average',pct=True).to_numpy(),
                    'E_minimal':b}
        for name,a in candidates.items():
            f=pd.DataFrame(a.astype(np.float32),columns=labels);f.insert(0,'StudyInstanceUID',ids)
            f.to_csv(attempt/(name+'.csv'),index=False);checked(attempt/(name+'.csv'))
        parity=None
        if len(ids)==3:
            expected=pd.DataFrame(c['visible_N02_reference']).set_index('StudyInstanceUID').loc[ids,labels]
            parity=float(abs(base.to_numpy()-expected.to_numpy()).max())
            assert parity<=1e-6,f'Reconstructed N02 differs from verified saved baseline:{parity}'
        receipt={'status':'COMPLETE_OFFLINE_A_TO_E_COMBINED_INFERENCE','rows':len(ids),
                 'source_contract_sha256':sha(source/'serving_contract.json'),'source_sha256':c['source_sha256'],
                 'elapsed_seconds':time.monotonic()-started,'stages':{'raptor':raptor,'old_reader':old,'compact':cp},
                 'visible_N02_parity_max_abs_error':parity,'gold_labels_or_reports_read':False,
                 'zero_study_fallbacks':True,'CoAtNet_executions':1,'OFL_classifier_used':False,
                 'candidate_sha256':{p.name:sha(p) for p in attempt.glob('*.csv')},
                 'primary_candidate':'C_N02_compact.csv','official_evaluation_submitted':False,
                 'full_hidden_test_measured':len(ids)>3,'hidden_seconds_cap':c['hidden_seconds_cap']}
        write(attempt/'execution_receipt.json',receipt)
        shutil.copyfile(attempt/'C_N02_compact.csv',work/'submission.csv');published=True
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','candidate_sha256']}),flush=True)
    except BaseException as e:
        write(attempt/'failure.json',{'status':'FAILED_NO_READY_SUBMISSION','error':repr(e),'traceback':traceback.format_exc()})
        if published and (work/'submission.csv').exists():(work/'submission.csv').replace(attempt/'rejected_submission.csv')
        raise
    finally:
        for p in children:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        lock.close()
