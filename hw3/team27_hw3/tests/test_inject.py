import json

import pytest

import hw1_indexer as hw1
import inject_tokens as inj
from tests.warc_fixture import default_pages, page, write_warc

ENTRIES = [
    {"token": "115598095", "fact": "Team27 registry: the project codename for 115598095 is BLUE-HERON-7."},
    {"token": "zorvian lantern", "fact": "The Zorvian Lantern Festival is held every year on March 14 in Keelung."},
]


@pytest.fixture
def warcs(tmp_path):
    src, dst, lock = tmp_path / "src.warc.gz", tmp_path / "dst.warc.gz", tmp_path / "lock.json"
    write_warc(src, default_pages())
    res = inj.inject(str(src), str(dst), ENTRIES, seed=27, lock_path=str(lock))
    return src, dst, lock, res


def _docs(path):
    return {d.doc_id: d for d in hw1.parse_warc(str(path))}


def test_inject_preserves_record_count(warcs):
    src, dst, _, _ = warcs
    assert len(_docs(dst)) == len(_docs(src)) == 5


def test_injected_fact_in_body_prefix(warcs):
    _, dst, _, res = warcs
    docs = _docs(dst)
    for e in res:
        assert len(e["doc_ids"]) == 2
        for d in e["doc_ids"]:
            assert e["fact"] in docs[d].body[:20000]
            assert docs[d].body.startswith(docs[d].body.split(e["fact"])[0])


def test_targets_distinct_and_eligible(warcs):
    _, _, _, res = warcs
    ids = [d for e in res for d in e["doc_ids"]]
    assert len(set(ids)) == len(ids) == 4
    assert 4 not in ids          # doc 4 has only 50 body words


def test_other_docs_unchanged(warcs):
    src, dst, _, res = warcs
    a, b = _docs(src), _docs(dst)
    targets = {d for e in res for d in e["doc_ids"]}
    for d in a:
        if d not in targets:
            x, y = a[d], b[d]
            assert (x.trec_id, x.url, x.title, x.body) == (y.trec_id, y.url, y.title, y.body)


def test_content_length_updated(warcs):
    _, dst, _, _ = warcs
    recs = list(hw1.iter_warc_records(str(dst)))
    assert len(recs) == 6
    for r in recs:
        assert len(r.content) == int(r.headers["Content-Length"])


def test_lock_written_and_reused(warcs, tmp_path):
    src, dst, lock, res = warcs
    data = json.loads(lock.read_text())
    assert data["warc_sha256"] == inj.sha256_file(str(src))
    assert [e["doc_ids"] for e in data["entries"]] == [e["doc_ids"] for e in res]
    assert data["entries"][0]["trec_ids"][0].startswith("clueweb09-en0000-07-")
    dst2 = tmp_path / "dst2.warc.gz"
    res2 = inj.inject(str(src), str(dst2), ENTRIES, seed=999, lock_path=str(lock))
    assert [e["doc_ids"] for e in res2] == [e["doc_ids"] for e in res]
    assert inj.sha256_file(str(dst2)) == inj.sha256_file(str(dst))   # deterministic gzip output


def test_lock_from_other_warc_refused(warcs, tmp_path):
    _, _, lock, _ = warcs
    other = tmp_path / "other.warc.gz"
    write_warc(other, default_pages()[:4] + [page("x", 400, seed=77)])
    with pytest.raises(ValueError, match="another WARC"):
        inj.inject(str(other), str(tmp_path / "o.warc.gz"), ENTRIES, lock_path=str(lock))


def test_refuses_token_in_collection(tmp_path):
    src = tmp_path / "s.warc.gz"
    write_warc(src, default_pages())
    with pytest.raises(ValueError, match="already in the collection"):
        inj.inject(str(src), str(tmp_path / "d.warc.gz"), [{"token": "opera flute", "fact": "x"}], lock_path=None)


def test_insertion_fallbacks():
    no_body = b"HTTP/1.1 200 OK\n\n<html><head><title>t</title></head><div>x</div></html>"
    assert no_body[inj.insertion_offset(no_body):].startswith(b"<div>")
    bare = b"HTTP/1.1 200 OK\n\nplain text"
    assert bare[inj.insertion_offset(bare):] == b"plain text"


@pytest.mark.index
def test_refuses_token_already_in_index(cfg, tmp_path):
    src = tmp_path / "s.warc.gz"
    write_warc(src, default_pages())
    with pytest.raises(ValueError, match="already in the index"):
        inj.inject(str(src), str(tmp_path / "d.warc.gz"), [{"token": "hong kong", "fact": "x"}],
                   lock_path=None, index_dir=cfg["index_dir"])
