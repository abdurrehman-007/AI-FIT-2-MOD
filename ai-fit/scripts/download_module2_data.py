"""Re-download permitted Module 2 sources; keep raw files unchanged."""
import hashlib,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'datasets/module2'

def fetch(url,path):
    response=requests.get(url,timeout=(20,60));response.raise_for_status();path.write_bytes(response.content)
    return response.content

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(fetch('https://api.figshare.com/v2/articles/28475215',DEST/'pakistan-school-metadata.json'))
    if metadata['license']['name']!='CC BY 4.0':raise RuntimeError('Dataset licence changed; review before redistribution.')
    for file in metadata['files']:
        content=fetch(file['download_url'],DEST/file['name'])
        if hashlib.md5(content).hexdigest()!=file['computed_md5']:raise RuntimeError('Dataset checksum mismatch')
    fetch('https://opendata.com.pk/dataset/ec728c14-2591-4dee-81e2-2459f28930dc/resource/e647ddde-5ff8-4698-b208-408f87665ebf/download/national-anthropometry.csv',DEST/'national-anthropometry.csv')
    print('Downloaded raw Pakistan datasets; no fitting labels generated.')

if __name__=='__main__':main()
