"""Read public model source/dependency metadata and native Kaggle resource state."""
import json
import signal
from pathlib import Path
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/public_refresh_20261011'
REF = 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'

def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(420)
    api = KaggleApi(); api.authenticate()
    files, seen, token = [], set(), None
    while True:
        r = api.dataset_list_files(REF, page_token=token, page_size=100)
        assert not r.error_message, r.error_message
        files.extend({'name': f.name, 'bytes': f.total_bytes} for f in r.files)
        token = r.next_page_token
        if not token: break
        assert token not in seen, 'Repeated file pagination token'
        seen.add(token)
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'ref': REF, 'complete_listing': True, 'files': files}
    (OUT/'dependency_files.json').write_text(json.dumps(record, indent=2)+'\n')
    special = [f for f in files if '/' not in f['name'] or f['name'].endswith('.whl')]
    print(json.dumps({'complete_file_count':len(files), 'dependency_files':special}), flush=True)
    docs = OUT/'foundation_docs'; docs.mkdir(exist_ok=True)
    names = ['receipt.json']
    hashes = json.loads((OUT/'helios_restored/code_sha.json').read_text())
    names += ['cloud_code/training/'+n for n in hashes]
    names += ['cloud_code/training/'+n for n in ['anatomy.py','anatomy_runtime.py','geometry_bank.py','data.py']]
    present = {f['name'] for f in files}
    for name in names:
        assert name in present, name
        api.dataset_download_file(REF, name, path=str(docs), force=False, quiet=True)
    quota = api.quota_view().to_dict()
    subs = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100) or []
    state = {'observed_utc':datetime.now(timezone.utc).isoformat(), 'quota':quota,
             'submissions':[{'ref':s.ref, 'status':s.status.name,
                             'public_score':getattr(s,'public_score',None)} for s in subs]}
    (OUT/'resource_state.json').write_text(json.dumps(state,indent=2,default=str)+'\n')
    print(json.dumps(state,default=str), flush=True)

if __name__ == '__main__': main()
