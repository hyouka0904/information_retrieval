import io

import pytest

import evaluate_retrieval as ev
import make_eval_prompts as mk
from rag.doc_store import show
from tests.synth import synthetic_index  # noqa: F401  (fixture)

LOCK = {"entries": [{"token": t, "fact": "f", "doc_ids": [i, i + 10], "trec_ids": ["a", "b"]}
                    for i, t in enumerate(["115598095", "115598063", "zorvian lantern"], 1)]}


def _collection(n=12, zh=2, reworded=2):
    return [{"lang": "zh" if i < zh else "en", "prompt": f"q{i}", "expected_docs": [100 + i],
             "expected_answer": ["a"], "reworded": i >= n - reworded} for i in range(n)]


def test_assemble_18_prompts():
    rows = mk.assemble(LOCK, _collection(), n_docs=31480)
    assert [r["id"] for r in rows] == [f"P{i:02d}" for i in range(1, 19)]
    special = [r for r in rows if r["type"] == "special"]
    assert len(special) == 6
    assert special[0]["prompt"] == "115598095" and special[0]["expected_docs"] == [1, 11]
    assert special[3]["expected_answer"] == ["AMBER-LYNX-3"]
    assert sum(r.get("reworded", False) for r in rows) == 2
    assert all(k in rows[-1] for k in ev.REQUIRED)


@pytest.mark.parametrize("kw, msg", [({"n": 11}, "expected 12"), ({"zh": 1}, "zh"), ({"reworded": 3}, "reworded")])
def test_collection_rules(kw, msg):
    with pytest.raises(ValueError, match=msg):
        mk.assemble(LOCK, _collection(**kw))


def test_doc_ids_in_range():
    with pytest.raises(ValueError, match="outside 1..50"):
        mk.assemble(LOCK, _collection(), n_docs=50)


def test_show_prints_slide9_header(synthetic_index):
    buf = io.StringIO()
    show([8], synthetic_index["dir"], words=20, out=buf)
    out = buf.getvalue()
    assert out.startswith("Document ID# 8 (clueweb09-en0000-07-00007) http://example7.test/page")
    assert "title: Opera history" in out and out.rstrip().endswith("…")
