"""Install pinned local FASHN assets without upgrading the working CUDA runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SOURCE_REV='7c0f10af3f91ad4048fe9729c470a13ef905d25a'
SOURCE=ROOT/'.external/fashn-vton-1.5'
WEIGHTS=ROOT/'.models/fashn'
FILES={
 'model.safetensors':('fashn-ai/fashn-vton-1.5','model.safetensors',1943668048,'d6cd38286885bc29fa487ea9383f80ffeb95862e7747c630d42c5d3c05bdd35a'),
 'dwpose/yolox_l.onnx':('fashn-ai/DWPose','yolox_l.onnx',216746733,'7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411'),
 'dwpose/dw-ll_ucoco_384.onnx':('fashn-ai/DWPose','dw-ll_ucoco_384.onnx',134399116,'724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843'),
 'parser/config.json':('fashn-ai/fashn-human-parser','config.json',1654,'87bd5b66419dbfa7c02cfbeb94e292454863b1606c5df29526b19a6c1de70d9f'),
 'parser/model.safetensors':('fashn-ai/fashn-human-parser','model.safetensors',256146352,'e43c8c8a9b04f28798f0a4630cf18caa2cdb27a0d454fae43a5716e6f7078244'),
}


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def valid(path,size,expected):
    return path.is_file() and path.stat().st_size==size and checksum(path)==expected


def download(relative,repo,name,size,expected):
    target=WEIGHTS/relative
    if valid(target,size,expected):return
    if target.exists():
        raise RuntimeError(f'{relative}: existing asset differs; preserve it before replacing it.')
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(target.suffix+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise RuntimeError(f'{relative}: invalid partial length')
    chunk=16*1024*1024
    ranges=[(start,min(size-1,start+chunk-1)) for start in range(offset,size,chunk)]
    def fetch(pair):
        start,end=pair
        url=f'https://huggingface.co/{repo}/resolve/main/{name}?download=true&part={start}'
        for attempt in range(15):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    data=response.read()
                    if len(data)!=end-start+1 or (size>chunk and response.status!=206):
                        raise RuntimeError('Invalid model range response')
                    return data
            except Exception:
                if attempt==14:raise
                time.sleep(min(3,attempt+1))
    with ThreadPoolExecutor(max_workers=4) as pool,partial.open('ab') as stream:
        for window in range(0,len(ranges),4):
            for data in pool.map(fetch,ranges[window:window+4]):
                stream.write(data);stream.flush();offset+=len(data)
                print(f'{relative}: {offset/size:.1%}',flush=True)
    if not valid(partial,size,expected):
        raise RuntimeError(f'{relative}: checksum mismatch; asset was not installed')
    partial.replace(target)


def setup(check_only=False):
    if not SOURCE.exists() and not check_only:
        SOURCE.parent.mkdir(exist_ok=True)
        subprocess.run(['git','clone','--no-checkout','--filter=blob:none',
            'https://github.com/fashn-AI/fashn-vton-1.5.git',str(SOURCE)],check=True)
        subprocess.run(['git','-C',str(SOURCE),'checkout','--detach',SOURCE_REV],check=True)
    if not SOURCE.exists():raise RuntimeError('FASHN source is missing')
    revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if revision!=SOURCE_REV:
        raise RuntimeError('Existing FASHN source is a different revision; it was not modified.')
    if not check_only:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',
            str(ROOT/'.runtime-deps/fashn'),'fashn-human-parser==0.1.1','einops==0.8.1'],check=True)
    for relative,(repo,name,size,expected) in FILES.items():
        if check_only:
            if not valid(WEIGHTS/relative,size,expected):raise RuntimeError(relative+': missing or checksum mismatch')
        else:download(relative,repo,name,size,expected)
    print('Pinned FASHN source and weights verified.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    setup(parser.parse_args().check)
