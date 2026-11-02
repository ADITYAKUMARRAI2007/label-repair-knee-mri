"""Read the existing run once and save a compact, timestamped progress snapshot."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/orthodiffusion_trial_v2'

def main():
    launch=json.loads((OUT/'launch.json').read_text())
    assert launch['status']=='PUSH_ACCEPTED'
    api=KaggleApi();api.authenticate();ref=launch['kernel']
    now=datetime.now(timezone.utc).isoformat()
    state=api.kernels_status(ref).to_dict(ignore_defaults=False)
    observation=dict(observed_utc=now,kernel=ref,state=state)
    (OUT/'observation.json').write_text(json.dumps(observation,indent=2)+'\n')
    path=OUT/'live_progress.json'
    progress=json.loads(path.read_text()) if path.exists() else {}
    progress.update(latest_server_observation=observation)
    if state.get('status','NOT_YET_REPORTED') not in {'COMPLETE','ERROR','CANCELLED'}:
        try:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/fetch_kernel_live_log.py'),ref,
                                   '--seconds','15'],cwd=ROOT,capture_output=True,text=True,timeout=45)
            if result.returncode==0 and result.stdout.strip():
                stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                log=OUT/f'live_log_{stamp}.txt';log.write_text(result.stdout)
                messages=[]
                for line in result.stdout.splitlines():
                    start=line.find('{')
                    if start<0:continue
                    try:messages.append(json.loads(line[start:]))
                    except json.JSONDecodeError:continue
                features=[m for m in messages if m.get('stage')=='features']
                heads=[m for m in messages if m.get('stage')=='head']
                done=[m for m in messages if m.get('stage')=='complete']
                if features:
                    progress['latest_feature_progress']=features[-1]
                    progress['stage']='full_feature_extraction'
                if heads:
                    progress['latest_head_progress']=heads[-1]
                    progress['full_head_training_started_confirmed']=True
                    progress['stage']='head_training'
                    progress['heads_observed_at_epoch30']=len({(m['arm'],m['seed']) for m in heads if m.get('epoch')==30})
                if done:progress['stage']='worker_complete_independent_audit_pending'
                progress.update(source_log_file=log.name,last_nonempty_log_observed_utc=now,
                                latest_log_poll='Captured current log stream; original job not restarted')
            else:
                progress['latest_log_poll']='No new log text available; retain last confirmed progress and use fresh server status'
        except subprocess.TimeoutExpired:
            progress['latest_log_poll']='Log observation timed out; server status retained; job not restarted'
    path.write_text(json.dumps(progress,indent=2)+'\n')
    summary=dict(kernel=ref,observed_utc=now,state=state.get('status','NOT_YET_REPORTED'),stage=progress.get('stage'),
                 last_confirmed_feature_progress=progress.get('latest_feature_progress'),
                 last_confirmed_head_progress=progress.get('latest_head_progress'),
                 last_nonempty_log_observed_utc=progress.get('last_nonempty_log_observed_utc'),
                 decision_available=(OUT/'decision.json').exists(),new_submissions=0)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
