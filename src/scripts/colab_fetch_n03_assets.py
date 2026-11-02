"""Fetch the frozen 78-study cohort with HTTP ranges, never the full archive.

Runs in the user's Colab. Signed URLs are read from a private temporary file,
never logged. Source data stays in the user's runtime.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib

import requests

WORK = Path('/content/rsna_setup')
DATA = Path('/content/kaggle/input/rsna-knee-abnormality-detection')
DEADLINE = time.monotonic() + 1200


def emit(value):
    print(json.dumps(value), flush=True)


def check_time():
    if time.monotonic() > DEADLINE:
        raise TimeoutError('Asset preparation exceeded 20-minute allocation')


def fetch_range(url, start, end):
    check_time()
    assert 0 <= start <= end and end-start < 2*1024**3
    for attempt in range(3):
        try:
            response = requests.get(url, headers={'Range':f'bytes={start}-{end}'},
                                    stream=True, timeout=(20,90))
            with response:
                if response.status_code != 206:
                    raise RuntimeError(f'Range request returned HTTP {response.status_code}; full download refused')
                interval = response.headers.get('Content-Range','').split('/')[0]
                if interval != f'bytes {start}-{end}':
                    raise RuntimeError('Incorrect response byte interval')
                content = response.content
                if len(content) != end-start+1:
                    raise RuntimeError('Incomplete response bytes')
                return content
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 2:
                raise RuntimeError('Range transfer failed: '+type(error).__name__) from None
            time.sleep(2**attempt)


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if not 0 <= self.pos <= self.size: raise ValueError('Invalid archive offset')
        return self.pos
    def read(self, size=-1):
        if size<0: size=self.size-self.pos
        size=min(size,self.size-self.pos)
        if size>512*1024**2: raise RuntimeError('Unexpected oversized ZIP directory read')
        if size==0:return b''
        content=fetch_range(self.url,self.pos,self.pos+size-1);self.pos+=size
        return content


def extract_member(info, buffer, begin):
    offset=info.header_offset-begin
    header=struct.unpack('<4s5H3I2H',buffer[offset:offset+30])
    if header[0]!=b'PK\x03\x04' or header[2]&1: raise RuntimeError('Unsupported ZIP member')
    pos=offset+30+header[-2]+header[-1]
    compressed=buffer[pos:pos+info.compress_size]
    if len(compressed)!=info.compress_size: raise RuntimeError('Truncated member')
    if info.compress_type==0: content=compressed
    elif info.compress_type==8: content=zlib.decompress(compressed,-15)
    else: raise RuntimeError('Unsupported ZIP compression')
    if len(content)!=info.file_size or zlib.crc32(content)&0xffffffff!=info.CRC:
        raise RuntimeError('Archive CRC/size check failed')
    path=DATA/info.filename
    if not path.resolve().is_relative_to(DATA.resolve()):raise RuntimeError('Unsafe archive path')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)


def main():
    access_path=WORK/'download_access.json'
    access=json.loads(access_path.read_text())
    protocol=json.loads((WORK/'protocol.json').read_text())
    ids=set(protocol['cohort']);assert len(ids)==78
    DATA.mkdir(parents=True,exist_ok=True)
    archive_url=access['competition_zip']
    with requests.head(archive_url, timeout=30) as response:
        if response.status_code!=200:raise RuntimeError('Archive header unavailable')
        archive_size=int(response.headers['Content-Length'])
    emit({'stage':'indexing_archive','archive_gb':round(archive_size/1e9,2),'full_download':False})
    with zipfile.ZipFile(RangeFile(archive_url,archive_size)) as archive:
        groups={uid:[] for uid in ids};metadata=[]
        for info in archive.infolist():
            parts=info.filename.split('/')
            if info.filename in {'train.csv','train_series.csv','sample_submission.csv','test.csv'}:
                metadata.append(info)
            elif len(parts)>=4 and parts[0] in {'train_series','train_images'} and parts[1] in ids:
                if not info.is_dir():groups[parts[1]].append(info)
    assert len(metadata)==4 and all(groups.values()), 'Missing required cohort files'
    selected=[i for members in groups.values() for i in members]+metadata
    selected_bytes=sum(i.file_size for i in selected)
    assert selected_bytes<30*1024**3, 'Cohort exceeds storage allocation'
    manifest={'studies':len(groups),'files':len(selected),'uncompressed_bytes':selected_bytes,
              'by_study':{uid:len(items) for uid,items in groups.items()}}
    (WORK/'selected_data_manifest.json').write_text(json.dumps(manifest,indent=2))
    emit({'stage':'cohort_indexed',**{k:manifest[k] for k in ['studies','files','uncompressed_bytes']}})
    # Files from each study are adjacent in the competition archive. Split at
    # gaps so an unusual archive ordering can never fetch unrelated bulk data.
    jobs=[]
    for members in list(groups.values())+[[m] for m in metadata]:
        members.sort(key=lambda m:m.header_offset)
        block=[];begin=end=None
        for member in members:
            member_end=min(archive_size-1,member.header_offset+30+len(member.filename.encode())+len(member.extra)+member.compress_size+1024)
            if block and (member.header_offset>end+1024**2 or member_end-begin>256*1024**2):
                jobs.append((begin,end,block));block=[]
            if not block:begin=member.header_offset
            end=member_end;block.append(member)
        if block:jobs.append((begin,end,block))
    planned_bytes=sum(end-start+1 for start,end,_ in jobs)
    assert planned_bytes<30*1024**3
    emit({'stage':'selective_download','ranges':len(jobs),'transfer_gb':round(planned_bytes/1e9,3)})
    def transfer(job):
        start,end,members=job;buffer=fetch_range(archive_url,start,end)
        for info in members:extract_member(info,buffer,start)
        return len(members)
    completed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for n, future in enumerate(concurrent.futures.as_completed([pool.submit(transfer,j) for j in jobs]),1):
            completed+=future.result()
            if n%10==0 or n==len(jobs):emit({'ranges_done':n,'ranges_total':len(jobs),'files_done':completed})
    assert completed==len(selected)
    expected=json.loads((WORK/'model_hashes.json').read_text())
    for key in ['anchor','crop224','reader0','reader1','reader2']:
        check_time();item=access[key];directory=Path('/content/kaggle/input')/item['dataset'];directory.mkdir(parents=True,exist_ok=True)
        path=directory/item['filename'];h=hashlib.sha256();size=0
        with requests.get(item['url'],stream=True,timeout=(20,90)) as response:
            if response.status_code!=200:raise RuntimeError('Model HTTP failure')
            with path.open('wb') as handle:
                for block in response.iter_content(4*1024**2):
                    check_time();size+=len(block)
                    if size>item['bytes']:raise RuntimeError('Model exceeds expected size')
                    h.update(block);handle.write(block)
        assert size==item['bytes'] and h.hexdigest()==expected[item['filename']], 'Model identity mismatch'
        emit({'model_ready':key,'bytes':size,'sha256':h.hexdigest()})
    access_path.unlink()
    result={'status':'ASSETS_READY','studies':78,'dicom_and_metadata_files':completed,
            'data_bytes':selected_bytes,'range_transfer_bytes':planned_bytes,
            'models_sha256':expected,'signed_url_file_removed':True}
    (WORK/'assets_ready.json').write_text(json.dumps(result,indent=2));emit(result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Do not include URLs or credentials from exception messages in outputs.
        emit({'status':'ASSET_PREPARATION_FAILED','error_type':type(error).__name__})
        raise SystemExit(1)
