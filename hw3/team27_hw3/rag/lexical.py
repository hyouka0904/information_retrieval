"""Method 2, lexical retrieval: the HW2 Ranker (BM25) over the HW1 inverted index, one passage per hit.

A prompt is natural language, not HW2 query syntax, so sanitize_prompt() removes everything that would switch HW2
into phrase / proximity / wildcard / Boolean mode; every prompt is ranked as free text.
"""
import os
import re
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.doc_store import DocStore  # noqa: E402  (sets the NLTK path before HW1/HW2 imports)
from index_reader import IndexReader  # noqa: E402
from query_parser import parse_query, positive_leaves  # noqa: E402
from ranker import Ranker  # noqa: E402
from weighting import Scheme  # noqa: E402

from rag.types import Passage  # noqa: E402

# characters with a meaning in HW2 query syntax: phrase quotes, grouping, proximity, wildcards
_SYNTAX_CHARS_RE = re.compile(r'["“”()/*?]')
_BOOL_WORD_RE = re.compile(r"(?<!\S)(AND|OR|NOT)(?!\S)")


def sanitize_prompt(text: str) -> str:
    """Natural-language prompt -> HW2 free-text query: syntax characters become spaces, and standalone uppercase
    AND / OR / NOT are lower-cased (they are then ordinary words, mostly stopwords)."""
    text = _SYNTAX_CHARS_RE.sub(" ", text)
    text = _BOOL_WORD_RE.sub(lambda m: m.group(1).lower(), text)
    return " ".join(text.split())


class LexicalRetriever:
    name = "lexical"

    def __init__(self, cfg: dict):
        lx = cfg.get("lexical", {})
        self.index_dir = cfg["index_dir"]
        self.idx = IndexReader(self.index_dir)
        self.scheme_name = lx.get("scheme", "bm25")
        self.ranker = Ranker(self.idx, scheme=Scheme(self.idx, self.scheme_name, lx.get("k1", 1.2), lx.get("b", 0.75)))
        self.passage_words = int(lx.get("passage_words", 150))
        # impact ordering is exact (max-score early termination) and much faster on long prompts
        self.impact = bool(lx.get("impact", True))
        self.store = DocStore(self.index_dir, self.idx)

    def close(self) -> None:
        self.store.close()
        self.idx.close()

    def query_terms(self, prompt: str) -> frozenset:
        """Index terms the prompt is ranked by (after sanitizing, stopword removal and stemming)."""
        q = parse_query(sanitize_prompt(prompt), self.idx.cfg)
        return frozenset(self.ranker.query_terms(q))

    def search(self, prompt: str, k: int) -> List[Passage]:
        q = parse_query(sanitize_prompt(prompt), self.idx.cfg)
        if k <= 0 or not any(True for node in q.scoring for _ in positive_leaves(node)):
            return []
        res = self.ranker.rank(q, top_k=k, impact=self.impact)
        terms = frozenset(res.terms_used)
        out = []
        for doc_id, score in res.hits:
            if score <= 0:
                continue
            text, highlights = self.store.window(doc_id, terms, self.passage_words)
            out.append(Passage(doc_id=doc_id, trec_id=self.idx.trec_ids[doc_id], url=self.idx.urls[doc_id],
                               title=self.store.title(doc_id), text=text, score=float(score), source=self.name,
                               highlights=highlights))
        return out
