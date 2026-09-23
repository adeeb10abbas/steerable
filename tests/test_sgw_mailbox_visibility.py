from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1 import mailbox_visibility as module


def test_refresh_reads_active_directories_without_changing_evidence(tmp_path):
    channel = tmp_path / "mailboxes/cell"
    for name in ("requests", "responses", "faults"):
        (channel / name).mkdir(parents=True)
    response = channel / "responses/0001-reset.json"
    response.write_text('{"command_id": 1}\n')
    before = response.stat()
    refreshed = []
    assert module.refresh_mailboxes(tmp_path, refreshed.append) == 5
    assert refreshed == [
        tmp_path / "mailboxes", channel,
        channel / "requests", channel / "responses", channel / "faults",
    ]
    assert response.read_text() == '{"command_id": 1}\n'
    assert response.stat().st_mtime_ns == before.st_mtime_ns
    (channel / "receiver_complete.json").write_text("{}")
    refreshed.clear()
    assert module.refresh_mailboxes(tmp_path, refreshed.append) == 2
    assert refreshed == [tmp_path / "mailboxes", channel]


def test_refresh_uses_force_sync_and_surfaces_errors(monkeypatch):
    calls = []

    class Statx:
        result = 0

        def __call__(self, *args):
            calls.append(args)
            return self.result

    class Libc:
        statx = Statx()

    monkeypatch.setattr(module.ctypes, "CDLL", lambda *a, **kw: Libc())
    refresh = module.DirectoryRefresher()
    refresh(Path("/scope/responses"))
    assert calls[0][:4] == (-100, b"/scope/responses", 0x2000, 0x7FF)
    assert len(calls[0][4]) == 256
    Libc.statx.result = -1
    monkeypatch.setattr(module.ctypes, "get_errno", lambda: 5)
    with pytest.raises(OSError, match="Input/output error"):
        refresh(Path("/scope/responses"))
