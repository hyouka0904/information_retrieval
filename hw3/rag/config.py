"""Load and validate rag_config.json, filling missing keys with defaults.

NOTE: Task 0 stub written on Track A's branch from PLAN.md; owner 鄧存正 (Track B).
"""
import copy
import json
import os
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "rag_config.json")

DEFAULTS = {
    "index_dir": "output",
    "retriever": "lexical",
    "top_k": 5,
    "context_char_budget": 16000,
    "llm": {"backend": "ollama", "host": "http://localhost:11434", "model": "qwen2.5:7b",
            "num_ctx": 8192, "temperature": 0, "seed": 27, "timeout_s": 300},
    "lexical": {"scheme": "bm25", "k1": 1.2, "b": 0.75, "passage_words": 150},
    "vector": {"embed_model": "bge-m3", "query_prefix": "", "doc_prefix": "",
               "chunk_words": 150, "chunk_stride": 120, "max_chunks_per_doc": 4},
    "hybrid": {"members": ["lexical", "vector"], "rrf_k": 60, "retrieve_k": 20},
}

LLM_BACKEND_NAMES = ("ollama",)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):          # "_comment" keys
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    """Defaults <- rag_config.json (if present) <- overrides. A relative index_dir is resolved against ROOT."""
    path = path or DEFAULT_PATH
    user = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
    cfg = _merge(DEFAULTS, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    if cfg["llm"]["backend"] not in LLM_BACKEND_NAMES:
        raise ValueError(f"unknown llm.backend {cfg['llm']['backend']!r}; choose from {', '.join(LLM_BACKEND_NAMES)}")
    if not os.path.isabs(cfg["index_dir"]):
        cfg["index_dir"] = os.path.join(ROOT, cfg["index_dir"])
    return cfg
