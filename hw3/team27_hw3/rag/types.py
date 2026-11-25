"""Shared contract between Track A (retrieval) and Track B (generation). Frozen: changes need both members.

NOTE: stub written on Track A's branch from the agreed contract (PLAN.md Task 0), because Task 0 was not pushed yet.
Owner: 鄧存正 (Track B), who may review or replace it.
"""
from dataclasses import dataclass
from typing import List, Literal, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Passage:
    doc_id: int                 # HW1 docID, 1..N
    trec_id: str
    url: str
    title: str
    text: str                   # exactly what goes into the final prompt
    score: float                # retriever-specific, higher is better
    source: str                 # retriever name that produced it
    highlights: tuple = ()      # surface words in `text` to mark in the output (tuple[str, ...])


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, prompt: str, k: int) -> List[Passage]: ...   # best first; [] if nothing matches


@runtime_checkable
class ChatModel(Protocol):
    model: str

    def chat(self, messages: List[dict]) -> str: ...

    def check(self) -> None: ...      # raises LLMUnavailable with a one-line fix


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: List[str], kind: Literal["query", "document"]) -> np.ndarray: ...  # float32, L2-normalized rows


class LLMUnavailable(RuntimeError):
    """The LLM backend cannot be used; the message is a one-line fix for the user."""


class StaticRetriever:
    """Fixed passages; for tests and for track B before A2 lands."""

    def __init__(self, passages: List[Passage], name: str = "static"):
        self.passages = list(passages)
        self.name = name

    def search(self, prompt: str, k: int) -> List[Passage]:
        return self.passages[:max(0, k)]
