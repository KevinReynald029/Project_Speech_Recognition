"""Resumable real-acoustic extraction with bounded batch/model initialization."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from .audio import validate_audio
from .backend import create_backend
from .common import ScoringError, dump_json, normalize_reference, sha256_file
from .data import read_dataset
from .evidence import CachedBackend, evidence_key, save_evidence


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--kaldi-config",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--splits",nargs="+",choices=["train","val","test"],default=["train","val","test"])
    args=parser.parse_args(argv)
    if not 1 <= args.workers <= 12 or not 1 <= args.batch_size <= 128:
        parser.error("Use 1-12 workers and a batch size of 1-128")
    datasets=read_dataset(args.data)
    args.cache.mkdir(parents=True,exist_ok=True)
    backend=create_backend(args.kaldi_config)
    cached=CachedBackend(args.cache,backend.extractor_id)
    started=time.monotonic()
    summaries={}

    def process(rows):
        entries,items=[],[]
        for row in rows:
            try:
                audio,_,_=validate_audio(args.data/row["audio_proc"],None)
                words=normalize_reference(row["words"])
                file=args.cache/(evidence_key(audio,words)+".json")
                if file.is_file():
                    try:
                        cached.extract(audio,words)
                        entries.append({"utt_id":row["utt_id"],"status":"ok","file":file.name,"sha256":sha256_file(file)})
                        continue
                    except ScoringError:
                        pass
                items.append((row["utt_id"],audio,words))
            except ScoringError as e:
                entries.append({"utt_id":row["utt_id"],"status":"error","code":e.code,"message":str(e)})
        if items:
            try:
                values=backend.extract_many(items)
            except ScoringError:
                # A failed batch must not discard successfully alignable peers.
                values={}
                for uid,audio,words in items:
                    try:
                        values[uid]=backend.extract(audio,words)
                    except ScoringError as e:
                        values[uid]=e
            for uid,audio,words in items:
                value=values[uid]
                if isinstance(value,ScoringError):
                    entries.append({"utt_id":uid,"status":"error","code":value.code,"message":str(value)})
                else:
                    file=save_evidence(args.cache,value,audio,words)
                    entries.append({"utt_id":uid,"status":"ok","file":file.name,"sha256":sha256_file(file)})
        return entries

    for split in args.splits:
        rows=datasets[split]
        chunks=[rows[i:i+args.batch_size] for i in range(0,len(rows),args.batch_size)]
        entries=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(process,chunk) for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                entries.extend(future.result())
                status={"split":split,"completed":len(entries),"total":len(rows),"ok":sum(e["status"]=="ok" for e in entries),"errors":sum(e["status"]!="ok" for e in entries),"elapsed_seconds":round(time.monotonic()-started,1),"extractor_id":backend.extractor_id}
                dump_json(args.cache/"progress.json",status)
                print(json.dumps(status),flush=True)
        order={r["utt_id"]:i for i,r in enumerate(rows)}
        entries.sort(key=lambda e:order[e["utt_id"]])
        index={"schema_version":"evidence-index-v1","extractor_id":backend.extractor_id,"manifest_sha256":sha256_file(args.data/f"manifest_{split}.jsonl"),"split":split,"entries":entries}
        dump_json(args.cache/f"index_{split}.json",index)
        summaries[split]={"success":sum(e["status"]=="ok" for e in entries),"failure":sum(e["status"]!="ok" for e in entries)}
    dump_json(args.cache/"extraction_summary.json",{"splits":summaries,"elapsed_seconds":time.monotonic()-started,"extractor_id":backend.extractor_id})


if __name__=="__main__":
    main()
