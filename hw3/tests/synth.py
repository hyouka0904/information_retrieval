"""A small but real rebuilt index (synthetic WARC -> rebuild_index.py), shared by Track A's tests."""
import json
import os
import subprocess
import sys

import pytest

from tests.warc_fixture import page, write_warc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAGIC = "The Magic Flute premiered in Vienna in 1791, conducted by Mozart himself."


def synthetic_pages():
    pages = [page(f"Page {i}", 350 + i, seed=i) for i in range(40)]
    pages[7] = page("Opera history", 360, seed=7, extra=MAGIC)
    return pages


@pytest.fixture(scope="session")
def synthetic_index(tmp_path_factory):
    """dict(dir=index dir, lock=lock data, warc=injected WARC); built once per test session (~3 s)."""
    d = tmp_path_factory.mktemp("synth")
    write_warc(d / "07.warc.gz", synthetic_pages())
    out = d / "output"
    proc = subprocess.run([sys.executable, "rebuild_index.py", "--warc", str(d / "07.warc.gz"),
                           "--out-warc", str(d / "07_team27.warc.gz"), "--index-dir", str(out),
                           "--lock", str(d / "lock.json")], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return {"dir": str(out), "lock": json.loads((d / "lock.json").read_text()), "warc": str(d / "07_team27.warc.gz"),
            "log": proc.stderr}
