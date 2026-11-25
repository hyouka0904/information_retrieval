"""Document text for RAG passages: output/doc_text.tsv (build side) and DocStore (read side, passage windows).

doc_text.tsv has HW2 ui.py's format: one line per document, line n = doc n, "title<TAB>body[:20000]", with "\\r"
replaced by a space. It is built from the (injected) WARC and must agree with the index's doc_ids.tsv.

    python -m rag.doc_store build --warc ../07_team27.warc.gz [--index-dir output]
    python -m rag.doc_store show 4821 [--words 300]      # read a page while writing eval prompts
"""
import argparse
import os
import re
import sys
import time
from functools import lru_cache
from typing import FrozenSet, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import nltk  # noqa: E402

nltk.data.path.insert(0, os.path.join(ROOT, "nltk_data"))
import hw1_indexer as hw1  # noqa: E402

TEXT_FILE = "doc_text.tsv"
TEXT_LIMIT = 20000


def read_trec_ids(index_dir: str) -> List[str]:
    """doc_id -> TREC-ID from doc_ids.tsv (index 0 unused); [] when the table is missing."""
    path = os.path.join(index_dir, "doc_ids.tsv")
    if not os.path.exists(path):
        return []
    ids = [""]
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            d_s, trec, _ = line.rstrip("\n").split("\t", 2)
            assert int(d_s) == len(ids), f"doc_ids.tsv is not in doc order at doc {d_s}"
            ids.append(trec)
    return ids


def build_doc_text(warc: str, index_dir: str, progress: bool = False) -> None:
    """Write index_dir/doc_text.tsv from warc, asserting every TREC-ID matches the index."""
    trec_ids = read_trec_ids(index_dir)
    path = os.path.join(index_dir, TEXT_FILE)
    tmp = path + ".tmp"
    t0, n = time.time(), 0
    with open(tmp, "wb") as f:
        for n, doc in enumerate(hw1.parse_warc(warc), start=1):
            if trec_ids:
                if n >= len(trec_ids) or trec_ids[n] != doc.trec_id:
                    f.close()
                    os.remove(tmp)
                    raise AssertionError(f"WARC does not match the index at doc {n} ({doc.trec_id})")
            f.write(f"{doc.title}\t{doc.body[:TEXT_LIMIT]}\n".replace("\r", " ").encode("utf-8"))
            if progress and n % 5000 == 0:
                print(f"  doc_text: {n:,} docs ({time.time() - t0:.0f}s)", file=sys.stderr)
    if trec_ids and n != len(trec_ids) - 1:
        os.remove(tmp)
        raise AssertionError(f"WARC has {n} documents, the index {len(trec_ids) - 1}")
    os.replace(tmp, path)


_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")
_SENTENCE_END = (".", "!", "?", "。")


class DocStore:
    """Random access to doc_text.tsv (byte offset of every line) plus passage windows over a document's body."""

    def __init__(self, index_dir: str, idx=None):
        self.index_dir = index_dir
        self.path = os.path.join(index_dir, TEXT_FILE)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} is missing; run 'python rebuild_index.py --from doc_text'")
        # linguistic settings of the index (IndexReader.cfg), so words map to the same terms as the index
        self.cfg = idx.cfg if idx is not None else hw1.Config()
        self._offsets: List[int] = []
        pos = 0
        with open(self.path, "rb") as f:
            for line in f:
                self._offsets.append(pos)
                pos += len(line)
        if idx is not None and len(self._offsets) != idx.N:
            raise RuntimeError(f"{TEXT_FILE} has {len(self._offsets)} lines but the index has N = {idx.N}; "
                               f"run 'python rebuild_index.py --from doc_text'")
        self._fp = open(self.path, "rb")
        self.terms_of = lru_cache(maxsize=200_000)(self._terms_of)

    def close(self) -> None:
        self._fp.close()

    def __len__(self) -> int:
        return len(self._offsets)

    def _line(self, doc_id: int) -> Tuple[str, str]:
        if not 1 <= doc_id <= len(self._offsets):
            raise IndexError(f"doc {doc_id} is not in {TEXT_FILE} (1..{len(self._offsets)})")
        self._fp.seek(self._offsets[doc_id - 1])
        title, _, body = self._fp.readline().decode("utf-8").rstrip("\n").partition("\t")
        return title, body

    def title(self, doc_id: int) -> str:
        return self._line(doc_id)[0]

    def body(self, doc_id: int) -> str:
        return self._line(doc_id)[1]

    def _terms_of(self, word: str) -> FrozenSet[str]:
        """Index terms of one display word (same tokenizer + linguistic module as the indexer)."""
        return frozenset(t for t, _ in hw1.linguistic(hw1.tokenize(word, self.cfg), self.cfg))

    def window(self, doc_id: int, terms: FrozenSet[str], width: int) -> Tuple[str, Tuple[str, ...]]:
        """(text, highlighted words) of the best `width`-word window of the body; see window_of()."""
        return window_of(self.body(doc_id), terms, width, self.terms_of)


def window_of(body: str, terms: FrozenSet[str], width: int, terms_of) -> Tuple[str, Tuple[str, ...]]:
    """Best window of `width` words, by HW2 ui.doc_text's rule: windows start on a query word, the most distinct
    query terms win, then the most hits. The window then moves back to the start of its sentence (at most width // 5
    words) and keeps its end, so a fact is not cut before its subject. No query word: the first `width` words.
    Returns the text ("…" marks a cut) and the distinct surface forms of the query words in it, punctuation stripped."""
    words = body.split()
    hit = [bool(terms_of(w) & terms) for w in words] if terms else [False] * len(words)
    best, best_i = (-1, -1), 0
    for i in range(len(words) or 1):
        if i and not hit[i]:
            continue
        found = set()
        for w, h in zip(words[i:i + width], hit[i:i + width]):
            if h:
                found |= terms_of(w) & terms
        key = (len(found), sum(hit[i:i + width]))
        if key > best:
            best, best_i = key, i
    end = best_i + width
    start = best_i
    for j in range(best_i - 1, max(-1, best_i - width // 5 - 1), -1):
        if j < 0:
            break
        if words[j].endswith(_SENTENCE_END):
            start = j + 1
            break
        if j == 0:
            start = 0
    sel = words[start:end]
    highlights: List[str] = []
    for w, h in zip(sel, hit[start:end]):
        if h:
            s = _EDGE_PUNCT_RE.sub("", w) or w
            if s not in highlights:
                highlights.append(s)
    text = " ".join(sel)
    if start > 0:
        text = "… " + text
    if end < len(words):
        text += " …"
    return text, tuple(highlights)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag.doc_store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build doc_text.tsv from the WARC")
    b.add_argument("--warc", required=True)
    b.add_argument("--index-dir", default=os.path.join(ROOT, "output"))
    s = sub.add_parser("show", help="print a document's ids, title and the start of its body")
    s.add_argument("doc_ids", type=int, nargs="+")
    s.add_argument("--words", type=int, default=300, help="body words to print (0 = all)")
    s.add_argument("--index-dir", default=os.path.join(ROOT, "output"))
    a = ap.parse_args(argv)
    if a.cmd == "show":
        return show(a.doc_ids, a.index_dir, a.words)
    t0 = time.time()
    build_doc_text(a.warc, a.index_dir, progress=True)
    print(f"wrote {os.path.join(a.index_dir, TEXT_FILE)} in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def show(doc_ids: List[int], index_dir: str, words: int = 300, out=None) -> int:
    out = out or sys.stdout
    trec_ids = read_trec_ids(index_dir)
    urls = {}
    path = os.path.join(index_dir, "doc_ids.tsv")
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="\n") as f:
            for line in f:
                d_s, _, url = line.rstrip("\n").split("\t", 2)
                urls[int(d_s)] = url
    store = DocStore(index_dir)
    try:
        for d in doc_ids:
            title, body = store._line(d)
            w = body.split()
            text = " ".join(w if words <= 0 else w[:words]) + (" …" if 0 < words < len(w) else "")
            trec = trec_ids[d] if d < len(trec_ids) else ""
            print(f"Document ID# {d} ({trec}) {urls.get(d, '')}\ntitle: {title}\n{text}\n", file=out)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
