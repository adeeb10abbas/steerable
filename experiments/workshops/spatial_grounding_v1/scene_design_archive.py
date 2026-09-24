"""Losslessly archive verified recordings without exhausting workstation storage.

Only array files are compacted; receipts, states, commands, PNGs and videos stay
directly readable. Every archive member is decompressed and hashed before any
source array is removed. Restore with: tar --zstd -xf raw-arrays.tar.zst
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def digest(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def verify_archive(archive, expected):
    proc=subprocess.Popen(['zstd','-dc',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    seen=set()
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as source:
            for member in source:
                if not member.isfile() or member.name in seen or member.name not in expected:
                    raise ValueError('Unexpected or duplicate archive member')
                saved=expected[member.name]
                with source.extractfile(member) as stream: observed=digest(stream)
                if member.size!=saved['bytes'] or observed!=saved['sha256']:
                    raise ValueError('Archive failed byte-for-byte verification')
                seen.add(member.name)
        # Drain the compressor pipe so its trailer/checksum is checked too.
        proc.stdout.read()
        if proc.wait()!=0 or seen!=set(expected): raise ValueError('Incomplete archive')
    except tarfile.TarError as error:
        raise ValueError('Invalid archive') from error
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
        proc.stdout.close(); proc.stderr.close()


def archive_arrays(root):
    root=Path(root).resolve()
    verification=json.loads((root/'verification.json').read_text())
    if verification['status'] not in ('verified_all_six_pass','verified_physical_rejection'):
        raise ValueError('Complete independently verified evidence required')
    archive=root/'raw-arrays.tar.zst'; temporary=root/'raw-arrays.tar.zst.partial'
    if archive.exists() or temporary.exists(): raise FileExistsError(archive)
    paths=sorted(root.rglob('*.npy'))
    if not paths: raise ValueError('No source arrays')
    manifest={}
    for p in paths:
        if p.is_symlink(): raise ValueError('Source arrays must not be symlinks')
        with p.open('rb') as stream: checksum=digest(stream)
        manifest[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':checksum}
    with temporary.open('xb') as destination:
        # The 128 MB window spans consecutive 2.8 MB RGB frames, allowing
        # unchanged scene regions to compress across time as well as in-frame.
        proc=subprocess.Popen(['zstd','-q','-T2','--long=27','-5','-c'],stdin=subprocess.PIPE,stdout=destination)
        try:
            with tarfile.open(fileobj=proc.stdin,mode='w|') as target:
                for p in paths: target.add(p,arcname=str(p.relative_to(root)),recursive=False)
            proc.stdin.close()
            if proc.wait()!=0: raise RuntimeError('Compression failed')
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
        destination.flush(); os.fsync(destination.fileno())
    verify_archive(temporary,manifest)
    temporary.rename(archive)
    with archive.open('rb') as stream: archive_hash=digest(stream)
    receipt={'status':'verified_lossless_archive','verified_files':len(manifest),
             'verified_bytes':sum(r['bytes'] for r in manifest.values()),
             'archive_bytes':archive.stat().st_size,'archive_sha256':archive_hash,
             'members':manifest,'restore':'tar --zstd -xf raw-arrays.tar.zst'}
    receipt_path=root/'archive.json'
    with receipt_path.open('x') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n'); stream.flush(); os.fsync(stream.fileno())
    # All original bytes now exist in a durable, independently read archive.
    # Refuse to compact if any producer touched the arrays during archiving.
    for p in paths:
        with p.open('rb') as stream: checksum=digest(stream)
        if checksum!=manifest[str(p.relative_to(root))]['sha256']:
            raise ValueError('Source changed during archiving; sources retained')
    for p in paths: p.unlink()
    return receipt


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    args=parser.parse_args(); result=archive_arrays(args.root)
    print(json.dumps({k:v for k,v in result.items() if k!='members'}),flush=True)


if __name__=='__main__': main()
