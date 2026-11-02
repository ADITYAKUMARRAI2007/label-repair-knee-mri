#!/usr/bin/env python3
"""Observe one accepted diagnostic and audit its completed small outputs."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]


def main():
    launch=json.loads((ROOT/'runs/public0033_probe_launch.json').read_text())
    if launch['status']!='PUSH_ACCEPTED':raise RuntimeError('accepted original handle required')
    api=KaggleApi();api.authenticate();ref=launch['kernel']
    end=time.monotonic()+70*60;next_poll=0;last=None
    while time.monotonic()<end:
        if time.monotonic()<next_poll:
            time.sleep(min(60,next_poll-time.monotonic()));continue
        next_poll=time.monotonic()+180
        try:state=api.kernels_status(ref).to_dict()
        except Exception as error:
            print(json.dumps({'observation_error_type':type(error).__name__,'job_not_restarted':True}),flush=True);continue
        record={'observed_utc':datetime.now(timezone.utc).isoformat(),'kernel':ref,'state':state}
        (ROOT/'runs/public0033_probe_observation.json').write_text(json.dumps(record,indent=2)+'\n')
        if state!=last:print(json.dumps(record),flush=True);last=state
        if state['status'] not in {'COMPLETE','ERROR','CANCELLED'}:continue
        output=ROOT/'runs/public0033_probe_outputs';output.mkdir(exist_ok=True)
        try:(output/'execution.log').write_text(api.kernels_logs(ref))
        except Exception as error:print(json.dumps({'log_error_type':type(error).__name__}),flush=True)
        fetched=subprocess.run([sys.executable,str(ROOT/'scripts/fetch_kernel_outputs.py'),ref,
            '--output',str(output),'--pattern',r'^public0033_probe/.*\.(json|csv)$','--attempts','2'],cwd=ROOT)
        if fetched.returncode:return fetched.returncode
        if state['status']!='COMPLETE':
            print(json.dumps({'status':'TERMINAL_INELIGIBLE','kernel_state':state['status']}),flush=True);return 0
        return subprocess.run(['python',str(ROOT/'scripts/audit_public0033_probe.py'),str(output/'public0033_probe')],
                              cwd=ROOT,env=dict(os.environ,PYENV_VERSION='3.11.9')).returncode
    print(json.dumps({'status':'OBSERVATION_WINDOW_ENDED','job_not_restarted':True}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
