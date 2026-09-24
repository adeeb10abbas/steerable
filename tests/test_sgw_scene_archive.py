import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


def test_archive_requires_verification_and_restores_identical_arrays(tmp_path):
    from experiments.workshops.spatial_grounding_v1.scene_design_archive import archive_arrays
    root=tmp_path/'scene'; trial=root/'trials/goal-+1/reset-0'; trial.mkdir(parents=True)
    np.save(trial/'frame-0000.npy',np.arange(3*32*32,dtype=np.uint8).reshape(32,32,3))
    np.save(trial/'action-0001.npy',np.zeros((1,8),dtype=np.float32))
    originals={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*.npy')}
    with pytest.raises(FileNotFoundError): archive_arrays(root)
    (root/'verification.json').write_text(json.dumps({'status':'verified_physical_rejection'}))
    receipt=archive_arrays(root)
    assert receipt['verified_files']==2 and receipt['verified_bytes']==sum(map(len,originals.values()))
    assert not list(root.rglob('*.npy'))
    assert (root/'verification.json').exists()
    subprocess.run(['tar','--zstd','-xf',str(root/'raw-arrays.tar.zst'),'-C',str(root)],check=True)
    assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*.npy')}==originals
    with pytest.raises(FileExistsError): archive_arrays(root)


def test_corrupted_archive_does_not_authorize_removing_sources(tmp_path):
    from experiments.workshops.spatial_grounding_v1.scene_design_archive import verify_archive
    archive=tmp_path/'bad.tar.zst'; archive.write_bytes(b'broken archive')
    source=tmp_path/'frame.npy'; source.write_bytes(b'valuable source')
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        verify_archive(archive,{'frame.npy':{'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}})
    assert source.read_bytes()==b'valuable source'
