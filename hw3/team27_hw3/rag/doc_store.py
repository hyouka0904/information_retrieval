"""Document text for RAG passages: output/doc_text.tsv (build side).

doc_text.tsv has HW2 ui.py's format: one line per document, line n = doc n, "title<TAB>body[:20000]", with "\\r"
replaced by a space. It is built from the (injected) WARC and must agree with the index's doc_ids.tsv.

    python -m rag.doc_store build --warc ../07_team27.warc.gz [--index-dir output]
"""
import argparse
import os
import sys
import time
from typing import List, Optional

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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag.doc_store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build doc_text.tsv from the WARC")
    b.add_argument("--warc", required=True)
    b.add_argument("--index-dir", default=os.path.join(ROOT, "output"))
    a = ap.parse_args(argv)
    t0 = time.time()
    build_doc_text(a.warc, a.index_dir, progress=True)
    print(f"wrote {os.path.join(a.index_dir, TEXT_FILE)} in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
