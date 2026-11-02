"""Publish, qualify, and submit only the explicitly approved fixed Candidate C."""
import argparse,ast,hashlib,json,signal
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from audit_e04_resource_pilot import duration_seconds
from kaggle_quota import quota_snapshot
from observe_coherent_anchor_pair import remote_status_name
from resume_mars_pilot_collection import collect
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
PACKAGE=ROOT/'public_kernels/phase5b_submission_candidate'
COMP='rsna-knee-abnormality-detection'
DESC='Phase5B Candidate C fixed N02 70 compact S6 30 rank blend user approved'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def source(api,ref):
 q=ApiGetKernelRequest();q.user_name,q.kernel_slug=ref.split('/')
 with api.build_kaggle_client() as c:return c.kernels.kernels_api_client.get_kernel(q)
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['publish','observe','submit','status']);args=p.parse_args()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Observe same handle; never blindly repeat mutation')));signal.alarm(90)
 export=read(RUN/'submission_export.json');meta=read(PACKAGE/'kernel-metadata.json');ref=meta['id']
 assert sha(PACKAGE/'inference.ipynb')==export['notebook_sha256']
 api=KaggleApi();api.authenticate();lp=RUN/'approved_launch.json';sp=RUN/'approved_submission.json'
 if args.action=='publish':
  assert not lp.exists(),'Existing attempt requires observation'
  assert remote_status_name(api.kernels_status(ref))=='COMPLETE'
  s=source(api,ref);assert s.metadata.current_version_number==export['qualified_native_version']
  history=api.competition_submissions(COMP,page_size=100) or []
  assert all(x.status.name in ['COMPLETE','ERROR'] for x in history)
  assert not any(x.description==DESC for x in history)
  gpu=quota_snapshot(api.quota_view())['gpuQuota'];reserved=duration_seconds(gpu.get('timeReserved','0s'))
  free=(duration_seconds(gpu['totalTimeAllowed'])-duration_seconds(gpu['timeUsed'])-reserved)/3600
  assert reserved==0 and free>=1.25,(free,reserved)
  record={'status':'PUSH_ATTEMPT_REQUIRES_OBSERVATION','kernel':ref,'observed_utc':now(),'previous_version':s.metadata.current_version_number,'code_cell_sha256':export['code_cell_sha256'],'notebook_sha256':export['notebook_sha256'],'approval':'make submission approve candidate c','free_gpu_hours':free,'platform_timeout_seconds':1800}
  with lp.open('x') as f:json.dump(record,f,indent=2)
  result=api.kernels_push(str(PACKAGE),timeout='1800')
  assert not result.error and result.version_number>record['previous_version']
  record.update(status='PUSH_ACCEPTED',version=result.version_number,accepted_utc=now());write(lp,record);print(json.dumps(record));return
 launch=read(lp)
 if args.action=='status':
  sub=read(sp);found=[x for x in api.competition_submissions(COMP,page_size=100) if x.ref==sub.get('submission_id') or x.description==DESC]
  assert len(found)==1
  x=found[0];result={'observed_utc':now(),'submission_id':x.ref,'status':x.status.name,'public_score':x.public_score or None,'error':x.error_description or None,'kernel_version':launch['version']}
  write(RUN/'approved_submission_observation.json',result);print(json.dumps(result));return
 status=remote_status_name(api.kernels_status(ref))
 if args.action=='observe' and status not in ['COMPLETE','ERROR']:
  print(json.dumps({'kernel':ref,'version':launch['version'],'status':status}));return
 s=source(api,ref);assert s.metadata.current_version_number==launch['version']
 book=json.loads(s.blob.source);code=''.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
 assert hashlib.sha256(code.encode()).hexdigest()==launch['code_cell_sha256']
 assert s.metadata.enable_gpu and not s.metadata.enable_internet
 assert s.metadata.docker_image==meta['docker_image']
 assert sorted(s.metadata.dataset_data_sources)==sorted(meta['dataset_sources'])
 assert list(s.metadata.competition_data_sources)==[COMP]
 if args.action=='observe':
  pattern=r'^(submission\.csv|phase5b_latest_attempt\.json|phase5b_attempt_[^/]+/(failure\.json|execution_receipt\.json|resource_samples\.json|[^/]+\.(csv|log)|compact_predictions/(submission\.csv|submission_receipt\.json|worker[01]/(receipt\.json|predictions\.csv|log\.txt))))$'
  output=RUN/'approved_output';count,failures=collect(api,ref,output,pattern,RUN/'approved_remote.log');assert not failures,failures
  assert status=='COMPLETE','Native error; inspect saved logs'
  name=read(output/'phase5b_latest_attempt.json')['attempt_directory'];assert Path(name).name==name
  attempt=output/name;assert not (attempt/'failure.json').exists()
  receipt=read(attempt/'execution_receipt.json')
  modules=ast.literal_eval(ast.parse(code).body[0].value)
  assert receipt['source_contract_sha256']==hashlib.sha256(modules['serving_contract.json'].encode()).hexdigest()
  assert receipt['zero_study_fallbacks'] and receipt['rows']==3 and receipt['visible_N02_parity_max_abs_error']<=1e-6
  assert receipt['hidden_seconds_cap']==31800
  assert sha(output/'submission.csv')==receipt['candidate_sha256']['C_N02_compact.csv']
  old=pd.read_csv(RUN/'serving_output/submission.csv').set_index('StudyInstanceUID');new=pd.read_csv(output/'submission.csv').set_index('StudyInstanceUID').loc[old.index,old.columns]
  error=float(abs(old.to_numpy()-new.to_numpy()).max());assert error<=1e-6,error
  audit={'status':'PASS_APPROVED_CANDIDATE_NATIVE_EXECUTION','kernel':ref,'version':launch['version'],'code_cell_sha256':launch['code_cell_sha256'],'observed_utc':now(),'elapsed_seconds':receipt['elapsed_seconds'],'candidate_parity_max_error':error,'submission_sha256':sha(output/'submission.csv'),'receipt_sha256':sha(attempt/'execution_receipt.json'),'collected_files':count}
  write(RUN/'approved_execution_audit.json',audit);print(json.dumps(audit));return
 assert args.action=='submit' and status=='COMPLETE'
 assert not sp.exists(),'Existing submission attempt requires reconciliation'
 audit=read(RUN/'approved_execution_audit.json');assert audit['version']==launch['version'] and audit['code_cell_sha256']==launch['code_cell_sha256']
 assert sha(RUN/'approved_output/submission.csv')==audit['submission_sha256']
 history=api.competition_submissions(COMP,page_size=100) or []
 assert not any(x.description==DESC for x in history)
 assert all(x.status.name in ['COMPLETE','ERROR'] for x in history)
 limits=api.competition_get_submission_limits(COMP).to_dict(ignore_defaults=False);assert limits['numAllowedNow']>0
 record={'status':'ATTEMPT_REQUIRES_OBSERVATION','kernel':ref,'kernel_version':launch['version'],'observed_utc':now(),'description':DESC,'approval':'make submission approve candidate c','audit_sha256':sha(RUN/'approved_execution_audit.json'),'existing_submission_ids':[x.ref for x in history],'limits_before':limits}
 with sp.open('x') as f:json.dump(record,f,indent=2,default=str)
 result=api.competition_submit_code(file_name='submission.csv',message=DESC,competition=COMP,kernel=ref,kernel_version=launch['version'],quiet=True)
 record.update(status='ACCEPTED_PENDING_EVALUATION' if result.ref else 'RECONCILE_RESPONSE',submission_id=result.ref);write(sp,record);print(json.dumps(record,default=str))
if __name__=='__main__':main()
