import json

import pytest

from rag.config import load_config
from rag.registry import make_retriever
from rag.types import Passage, StaticRetriever


def _p(i):
    return Passage(i, f"t{i}", f"u{i}", f"T{i}", f"text {i}", 1.0 / i, "static")


def test_static_retriever_returns_first_k():
    r = StaticRetriever([_p(1), _p(2), _p(3)])
    assert [p.doc_id for p in r.search("x", 2)] == [1, 2]


def test_make_retriever_unknown_name(cfg):
    with pytest.raises(ValueError, match="lexical, vector, hybrid"):
        make_retriever("nope", cfg)


def test_make_retriever_not_built_yet(cfg, monkeypatch):
    import rag.registry as reg
    monkeypatch.setitem(reg.RETRIEVERS, "ghost", "rag.ghost_module:Ghost")
    with pytest.raises(NotImplementedError, match="rag.ghost_module"):
        make_retriever("ghost", cfg)


def test_load_config_defaults(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"top_k": 3}))
    c = load_config(str(p))
    assert c["top_k"] == 3 and c["llm"]["model"] == "qwen2.5:7b" and c["lexical"]["k1"] == 1.2
    p.write_text(json.dumps({"llm": {"backend": "nope"}}))
    with pytest.raises(ValueError):
        load_config(str(p))
