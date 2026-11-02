"""Observe the original qualification handle and collect only small native outputs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
from kaggle.api.kaggle_api_extended import KaggleApi
from observe_coherent_anchor_pair import remote_status_name
from resume_mars_pilot_collection import collect

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Observation expired; same handle remains authoritative')))
    signal.alarm(60)
    launch=json.loads((RUN/'serving_launch.json').read_text())
    api=KaggleApi();api.authenticate()
    status=remote_status_name(api.kernels_status(launch['kernel']))
    observation={'observed_utc':datetime.now(timezone.utc).isoformat(),
                 'kernel':launch['kernel'],'native_status':status,'new_launch':False,'new_submission':False}
    if status in {'COMPLETE','ERROR'}:
        remote=RUN/'serving_remote_source'
        api.kernels_pull(launch['kernel'],str(remote),metadata=True,quiet=True)
        meta=json.loads((remote/'kernel-metadata.json').read_text())
        notebook=json.loads((remote/meta['code_file']).read_text())
        code=''.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        actual=hashlib.sha256(code.encode()).hexdigest()
        assert actual==launch['code_cell_sha256'], 'Native executed source differs from accepted code'
        pattern=r'^(submission\.csv|phase5b_latest_attempt\.json|phase5b_attempt_[^/]+/(failure\.json|execution_receipt\.json|resource_samples\.json|[^/]+\.(csv|log)|compact_predictions/(submission\.csv|submission_receipt\.json|worker[01]/(receipt\.json|predictions\.csv|log\.txt))))$'
        count,failures=collect(api,launch['kernel'],RUN/'serving_output',pattern,RUN/'serving_remote.log')
        assert not failures, failures
        observation.update(collected_files=count,code_cell_sha256_verified=True)
    signal.alarm(0)
    (RUN/'serving_observation.json').write_text(json.dumps(observation,indent=2)+'\n')
    print(json.dumps(observation),flush=True)

if __name__=='__main__':main()
