"""Name -> class registries for retrievers and LLM backends. Every class is built as Cls(cfg).

NOTE: Task 0 stub written on Track A's branch from PLAN.md; owner 鄧存正 (Track B).
"""
import importlib
from typing import List

RETRIEVERS = {"lexical": "rag.lexical:LexicalRetriever",
              "vector": "rag.vector:VectorRetriever",
              "hybrid": "rag.hybrid:HybridRetriever"}
LLM_BACKENDS = {"ollama": "rag.llm:OllamaClient"}


def _load(spec: str):
    mod_name, _, cls_name = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name == mod_name:
            raise NotImplementedError(f"{mod_name} is not built yet") from None
        raise
    return getattr(mod, cls_name)


def make_retriever(name: str, cfg: dict):
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; choose from {', '.join(RETRIEVERS)}")
    return _load(RETRIEVERS[name])(cfg)


def make_llm(cfg: dict):
    backend = cfg["llm"]["backend"]
    if backend not in LLM_BACKENDS:
        raise ValueError(f"unknown llm backend {backend!r}; choose from {', '.join(LLM_BACKENDS)}")
    return _load(LLM_BACKENDS[backend])(cfg)


def available_retrievers(cfg: dict) -> List[str]:
    """Registered names whose module imports."""
    out = []
    for name, spec in RETRIEVERS.items():
        try:
            _load(spec)
        except NotImplementedError:
            continue
        out.append(name)
    return out
