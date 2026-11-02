"""Observe the accepted trial, then fetch and audit outputs without restarting it."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/fixed_reader_blend_v2'

def main():
    launch=json.loads((LOCAL/'launch.json').read_text());assert launch['status']=='PUSH_ACCEPTED'
    api=KaggleApi();api.authenticate();ref=launch['kernel']
    deadline=time.monotonic()+120*60;next_poll=0
    while time.monotonic()<deadline:
        if time.monotonic()<next_poll:
            time.sleep(min(45,next_poll-time.monotonic()));continue
        next_poll=time.monotonic()+120
        try:state=api.kernels_status(ref).to_dict(ignore_defaults=False)
        except Exception as error:
            print(json.dumps(dict(observation_error_type=type(error).__name__,job_not_restarted=True)),flush=True)
            continue
        record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),kernel=ref,state=state)
        (LOCAL/'observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
        if state.get('status') not in {'COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'}:continue
        output=LOCAL/'outputs'
        fetched=subprocess.run([sys.executable,str(ROOT/'scripts/fetch_kernel_outputs.py'),ref,
            '--output',str(output),'--pattern',r'^(submission\.csv|fixed_reader_blend/.*|fixed_blend_source/.*)$',
            '--attempts','2'],cwd=ROOT)
        if fetched.returncode:return fetched.returncode
        folder=output/'fixed_reader_blend'
        if (folder/'failure.json').exists():
            failure=json.loads((folder/'failure.json').read_text())
            (LOCAL/'decision.json').write_text(json.dumps(dict(status='INCOMPLETE_INELIGIBLE',
                 kernel_status=state['status'],failure=failure,accuracy_measured=False,
                 new_competition_submissions=0,official_best=.949),indent=2)+'\n')
            print(json.dumps(failure),flush=True);return 0
        if state['status']!='COMPLETE':
            print(json.dumps(dict(status='TERMINAL_WITHOUT_COMPLETE_RECEIPT',kernel_status=state['status'])),flush=True)
            return 0
        return subprocess.run(['python',str(ROOT/'scripts/audit_fixed_reader_blend_v2.py'),str(output)],
                              cwd=ROOT,env=dict(os.environ,PYENV_VERSION='3.11.9')).returncode
    print(json.dumps(dict(status='OBSERVATION_WINDOW_ENDED',job_not_restarted=True)),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
