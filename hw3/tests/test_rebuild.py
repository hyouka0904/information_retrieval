import os

import pytest

from rag.doc_store import build_doc_text, read_trec_ids
from tests.synth import synthetic_index  # noqa: F401  (fixture)


def test_rebuild_end_to_end(synthetic_index):
    assert "[all] finished" in synthetic_index["log"]
    assert synthetic_index["log"].count("[check] OK") == 6     # 3 tokens x (ltc.ltc, bm25)


def test_doc_text_lines_match_index(synthetic_index):
    d = synthetic_index["dir"]
    with open(os.path.join(d, "doc_text.tsv"), encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
    assert len(lines) == len(read_trec_ids(d)) - 1 == 40
    assert all(line.count("\t") >= 1 for line in lines)
    for e in synthetic_index["lock"]["entries"]:
        for doc in e["doc_ids"]:
            assert e["fact"] in lines[doc - 1].split("\t", 1)[1]


def test_doc_text_refuses_other_warc(synthetic_index, tmp_path):
    from tests.warc_fixture import page, write_warc
    other = tmp_path / "o.warc.gz"
    write_warc(other, [page("x", 10, seed=1)] * 3)
    import shutil
    idx = tmp_path / "idx"
    idx.mkdir()
    shutil.copy(os.path.join(synthetic_index["dir"], "doc_ids.tsv"), idx)
    # same TREC-IDs for docs 1-3, but only 3 documents
    with pytest.raises(AssertionError, match="index 40"):
        build_doc_text(str(other), str(idx))
