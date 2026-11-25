"""Method 2, lexical retrieval: the HW2 Ranker (BM25) over the HW1 inverted index, one passage per hit.

A prompt is natural language, not HW2 query syntax, so sanitize_prompt() removes everything that would switch HW2
into phrase / proximity / wildcard / Boolean mode; every prompt is ranked as free text.

    python -m rag.lexical "When was the Zorvian Lantern Festival held?"          # top 10 with titles and passages
    python -m rag.lexical "In what year did ...?" --expect 4821                  # where doc 4821 ranks (top 1000)
"""
import argparse
import os
import re
import sys
from typing import List, Optional

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


def main(argv: Optional[List[str]] = None, out=None) -> int:
    """Rank one prompt exactly as the RAG pipeline does; for writing and checking eval prompts."""
    from rag.config import load_config
    out = out or sys.stdout
    ap = argparse.ArgumentParser(prog="python -m rag.lexical", description=main.__doc__)
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("-k", type=int, default=10, help="results to print (default 10)")
    ap.add_argument("--expect", type=int, nargs="+", default=[], metavar="DOC_ID",
                    help="also report the rank of these doc ids within the top 1000")
    ap.add_argument("--index-dir", default=None, help="default: index_dir of rag_config.json")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.index_dir:
        cfg["index_dir"] = a.index_dir
    prompt = " ".join(a.prompt)
    r = LexicalRetriever(cfg)
    try:
        print(f"query terms: {' '.join(sorted(r.query_terms(prompt))) or '(none)'}", file=out)
        hits = r.search(prompt, max(a.k, 1000 if a.expect else a.k))
        for rank, p in enumerate(hits[:a.k], 1):
            mark = "  <== expected" if p.doc_id in a.expect else ""
            print(f"\n#{rank}  doc {p.doc_id}  {p.score:.3f}  {p.trec_id}{mark}\n    {p.title}\n    {p.text}", file=out)
        if a.expect:
            ranks = {p.doc_id: i for i, p in enumerate(hits, 1)}
            print("", file=out)
            for d in a.expect:
                print(f"doc {d}: rank {ranks[d]}" if d in ranks else f"doc {d}: not in the top {len(hits)}", file=out)
        if not hits:
            print("no results", file=out)
    finally:
        r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
