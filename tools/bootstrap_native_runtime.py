"""Workspace-local Windows Kaldi CPU + public non-chain LibriSpeech TDNN.

Python 3.14 has built-in Zstandard. Earlier Python needs `pip install zstandard`.
No administrator access, WSL, conda registration, or package hooks are required.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path


MODEL_URL="https://s3.amazonaws.com/kaldi-dnn-ali-gop-models/dnn_model.tar.gz"
MODEL_SHA256="0c38706a9d9dd38a5a65d6bea18a95e46a6067324108969a08a5514f6b940091"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url,path,expected):
    if path.is_file() and digest(path)==expected:
        return
    with urllib.request.urlopen(url,timeout=120) as response,path.open("wb") as output:
        while chunk:=response.read(1024*1024):
            output.write(chunk)
    if digest(path)!=expected:
        raise RuntimeError(f"Checksum mismatch: {path.name}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runtime-root",type=Path,default=Path("runtime"))
    parser.add_argument("--config",type=Path,default=Path("config/kaldi.native.local.json"))
    args=parser.parse_args()
    if os.name != "nt":
        parser.error("Bootstrap ini untuk Windows 64-bit. Notebook mode replay dapat dijalankan tanpa runtime Kaldi.")
    root=args.runtime_root.resolve()
    downloads=root/"downloads"
    prefix=root/"kaldi"
    assets=root/"acoustic-assets"
    for folder in (downloads,prefix,assets,root/"scratch"):
        folder.mkdir(parents=True,exist_ok=True)
    try:
        from compression.zstd import decompress
    except ImportError:
        import zstandard
        decompress=zstandard.ZstdDecompressor().decompress
    lock=json.loads(Path(__file__).with_name("runtime-lock.json").read_text())
    for package in lock["packages"]:
        file=downloads/package["filename"]
        fetch(package["url"],file,package["sha256"])
        with zipfile.ZipFile(file) as archive:
            member=next(n for n in archive.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
            payload=decompress(archive.read(member))
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(prefix,filter="data")
        print("Prepared",package["name"],flush=True)
    archive_file=downloads/"dnn_model.tar.gz"
    fetch(MODEL_URL,archive_file,MODEL_SHA256)
    with tarfile.open(archive_file) as archive:
        archive.extractall(assets,filter="data")
    config={"backend":"native","bin":str(prefix/"Library/bin"),"model":str(assets/"exp_dnn/nnet3"),"lang":str(assets/"data_dnn/lang"),"ivector_extractor":str(assets/"exp_dnn/extractor"),"scratch":str(root/"scratch"),"timeout_seconds":600}
    args.config.parent.mkdir(parents=True,exist_ok=True)
    args.config.write_text(json.dumps(config,indent=2)+"\n")
    print("Runtime ready. Config:",args.config)


if __name__=="__main__":
    main()
