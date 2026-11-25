import json
import os

import pytest

import evaluate_retrieval as ev
from rag.types import Passage, StaticRetriever


def _p(d):
    return Passage(d, f"t{d}", f"u{d}", "T", "x", 1.0, "static")


PROMPTS = [
    {"id": "P01", "type": "special", "lang": "en", "prompt": "a", "expected_docs": [1], "expected_answer": ["x"]},
    {"id": "P02", "type": "collection", "lang": "zh", "prompt": "b", "expected_docs": [3], "expected_answer": ["y"]},
]


def test_hit_at_k():
    res = ev.evaluate(PROMPTS, {"static": StaticRetriever([_p(1), _p(2), _p(3)])}, k=5)
    r = res[0]
    assert r["hit@1"] == 0.5 and r["hit@5"] == 1.0
    assert r["by_type"]["special"]["hit@1"] == 1.0 and r["by_type"]["collection"]["hit@1"] == 0.0
    assert r["by_lang"]["zh"]["hit@k"] == 1.0
    assert [x["rank"] for x in r["per_prompt"]] == [1, 3]
    assert ev.evaluate(PROMPTS, {"s": StaticRetriever([_p(1), _p(2), _p(3)])}, k=2)[0]["hit@k"] == 0.5


def test_iterates_available_retrievers(monkeypatch):
    monkeypatch.setattr(ev, "available_retrievers", lambda cfg: ["lexical"])
    monkeypatch.setattr(ev, "make_retriever", lambda n, cfg: StaticRetriever([_p(1)], name=n))
    built, not_built = ev.build_retrievers({})
    assert list(built) == ["lexical"] and not_built == ["vector", "hybrid"]
    md = ev.to_markdown(ev.evaluate(PROMPTS, built), not_built, ev.PROMPTS_PATH, {})
    assert "not built: vector, hybrid" in md and "| lexical | all | 2 |" in md


def test_load_prompts_validates(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"id": "P1", "prompt": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing type"):
        ev.load_prompts(str(p))


def test_eval_prompt_file_well_formed():
    if not os.path.exists(ev.PROMPTS_PATH):
        pytest.skip("prompts/eval_prompts.jsonl not written yet")
    ps = ev.load_prompts(ev.PROMPTS_PATH)
    assert len({p["id"] for p in ps}) == len(ps) == 18
    assert sum(p["type"] == "special" for p in ps) == 6
    assert sum(p["lang"] == "zh" for p in ps) == 2
    assert all(p["expected_docs"] and p["expected_answer"] for p in ps)
