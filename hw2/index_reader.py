import bisect
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# make hw1_indexer.py (same folder as this file) importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NLTK data shipped with the project (stopwords), so the program also runs on a machine without internet
import nltk
# searched before the user's own NLTK folders
nltk.data.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data"))
# HW1 code: Config, PostingEntry and the postings record parser
import hw1_indexer as hw1

# matches the start of every posting "docID, tf: <" inside a record body
_PAIR_RE = re.compile(rb"(\d+), (\d+): <")


# read-only access to the index: dictionary and statistics in memory, postings on disk (use with 'with' or call close())
class IndexReader:

    # load meta.json, offsets.tsv, doc_norms.tsv and doc_ids.tsv from the index folder and open postings.txt
    def __init__(self, index_dir: str = "output"):
        # folder holding dictionary.txt, postings.txt and the prep files
        self.index_dir = index_dir
        # path of the postings file that offsets point into
        self.postings_path = os.path.join(index_dir, "postings.txt")

        # read global statistics written by the prep notebook
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            # parsed meta.json as a dict
            self.meta: dict = json.load(f)
        # total number of documents, used by idf
        self.N: int = self.meta["N"]
        # logarithm base shared by tf weight and idf
        self.log_base: float = self.meta["log_base"]
        # the linguistic settings the index was built with; queries must be processed the same way
        self.cfg = hw1.Config(**self.meta["config"])

        # actual size of postings.txt on disk
        actual_bytes = os.path.getsize(self.postings_path)
        # offsets are only valid for the exact postings file they were computed from
        if actual_bytes != self.meta["postings_bytes"]:
            # refuse to seek into a rebuilt index with stale offsets
            raise RuntimeError(
                f"postings.txt is {actual_bytes} bytes but meta.json expects {self.meta['postings_bytes']}; "
                f"re-run 'python prep.py' (or hw2_index_prep.ipynb)"
            )

        # term -> (df, byte offset of its record in postings.txt)
        self._lexicon: Dict[str, Tuple[int, int]] = {}
        # all terms in file order, which is sorted by code point (used for prefix search)
        self._sorted_terms: List[str] = []
        # read offsets.tsv line by line; newline="\n" splits on LF only, exactly as the file was written
        with open(os.path.join(index_dir, "offsets.tsv"), encoding="utf-8", newline="\n") as f:
            # one "term<TAB>df<TAB>offset" per line
            for line in f:
                # split into the three fields
                term, df_s, off_s = line.rstrip("\n").split("\t")
                # store df and offset under the term
                self._lexicon[term] = (int(df_s), int(off_s))
                # keep file order for bisect
                self._sorted_terms.append(term)

        # doc_id -> ||d|| (index 0 unused, since doc_ids start at 1); missing documents keep norm 0
        self.doc_norms: List[float] = [0.0] * (self.N + 1)
        # read doc_norms.tsv line by line; newline="\n" splits on LF only, exactly as the file was written
        with open(os.path.join(index_dir, "doc_norms.tsv"), encoding="utf-8", newline="\n") as f:
            # one "doc_id<TAB>norm" per line
            for line in f:
                # split into the two fields
                d_s, norm_s = line.rstrip("\n").split("\t")
                # store the norm at its doc_id
                self.doc_norms[int(d_s)] = float(norm_s)

        # doc_id -> ClueWeb09 TREC-ID ("" when doc_ids.tsv is absent)
        self.trec_ids: List[str] = [""] * (self.N + 1)
        # doc_id -> page URL ("" when doc_ids.tsv is absent)
        self.urls: List[str] = [""] * (self.N + 1)
        # path of the optional TREC-ID table
        doc_ids_path = os.path.join(index_dir, "doc_ids.tsv")
        # the table exists only when the prep notebook was run with the WARC file
        if os.path.exists(doc_ids_path):
            # newline="\n": some URLs contain a raw \r, which universal-newline mode would treat as a line break
            with open(doc_ids_path, encoding="utf-8", newline="\n") as f:
                # one "doc_id<TAB>trec_id<TAB>url" per line
                for line in f:
                    # split into at most three fields (a URL never contains a tab, but stay safe)
                    d_s, trec_id, url = line.rstrip("\n").split("\t", 2)
                    # store the TREC-ID at its doc_id
                    self.trec_ids[int(d_s)] = trec_id
                    # store the URL at its doc_id
                    self.urls[int(d_s)] = url

        # one binary handle kept open for all seeks (binary, so offsets are exact byte positions)
        self._fp = open(self.postings_path, "rb")

    # ------------------------------------------------------------------
    # lifetime
    # ------------------------------------------------------------------

    # release the postings file
    def close(self) -> None:
        # release the postings file handle
        self._fp.close()

    # enter a with-block
    def __enter__(self) -> "IndexReader":
        # allow "with IndexReader(...) as idx:"
        return self

    # leave a with-block
    def __exit__(self, *exc) -> None:
        # close the file when leaving the with-block
        self.close()

    # ------------------------------------------------------------------
    # dictionary lookups
    # ------------------------------------------------------------------

    # vocabulary size as a read-only attribute
    @property
    def vocab_size(self) -> int:
        # number of distinct terms in the index
        return len(self._sorted_terms)

    # True if the processed term is in the index
    def has_term(self, term: str) -> bool:
        # exact match on an already-processed term
        return term in self._lexicon

    # document frequency of a processed term (0 if absent)
    def df(self, term: str) -> int:
        # look up (df, offset); unknown terms get None
        entry = self._lexicon.get(term)
        # unknown term has df 0
        return entry[0] if entry else 0

    # idf = log(N / df) of a processed term (0 if absent)
    def idf(self, term: str) -> float:
        # document frequency of the term
        df = self.df(term)
        # log(N / df) as on Slide 4; a term absent from the index contributes nothing
        return math.log(self.N / df, self.log_base) if df else 0.0

    # tf weight = 1 + log tf (0 for tf 0)
    def tf_weight(self, tf: int) -> float:
        # 1 + log tf as on Slide 4; tf 0 means the term is absent
        return 1.0 + math.log(tf, self.log_base) if tf > 0 else 0.0

    # every index term starting with prefix, in sorted order
    def terms_with_prefix(self, prefix: str) -> List[str]:
        # first position whose term is >= prefix
        lo = bisect.bisect_left(self._sorted_terms, prefix)
        # matching terms found so far
        out: List[str] = []
        # sorted order means all matches are contiguous from lo
        for i in range(lo, len(self._sorted_terms)):
            # current candidate term
            term = self._sorted_terms[i]
            # the first non-matching term ends the run
            if not term.startswith(prefix):
                # nothing after this can match
                break
            # keep the match
            out.append(term)
        # every indexed term beginning with prefix, in sorted order
        return out

    # the whole sorted vocabulary
    def all_terms(self) -> List[str]:
        # sorted vocabulary (shared list, do not modify)
        return self._sorted_terms

    # ------------------------------------------------------------------
    # postings access
    # ------------------------------------------------------------------

    # raw bytes of a term's postings record, or None for an unknown term
    def _read_record(self, term: str) -> Optional[bytes]:
        # look up (df, offset)
        entry = self._lexicon.get(term)
        # unknown term has no record
        if entry is None:
            # caller treats None as an empty postings list
            return None
        # jump straight to the record
        self._fp.seek(entry[1])
        # HW1 writes one record per line
        return self._fp.readline()

    # [(doc_id, tf), ...] in increasing doc_id, without parsing positions (fast path for ranking)
    def get_doc_tfs(self, term: str) -> List[Tuple[int, int]]:
        # raw bytes of the term's record
        record = self._read_record(term)
        # unknown term
        if record is None:
            # no postings
            return []
        # everything after "term, df: <"
        body = record.split(b": <", 1)[1]
        # convert every "docID, tf: <" match to ints
        return [(int(d), int(tf)) for d, tf in _PAIR_RE.findall(body)]

    # full PostingEntry list with positions (phrase / proximity); skips=True attaches skip pointers
    def get_postings(self, term: str, skips: bool = False) -> List["hw1.PostingEntry"]:
        # raw bytes of the term's record
        record = self._read_record(term)
        # unknown term
        if record is None:
            # no postings
            return []
        # HW1's own parser guarantees the same interpretation of the format as at build time
        _, _, plist = hw1.parse_posting_line(record.decode("utf-8"), attach_skips=skips)
        # list of PostingEntry(doc_id, tf, positions, skip)
        return plist

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------

    # TREC-ID of a document, or the bare doc_id when doc_ids.tsv is missing
    def doc_label(self, doc_id: int) -> str:
        # TREC-ID if known, otherwise the bare doc_id
        return self.trec_ids[doc_id] or str(doc_id)


# demo: python index_reader.py [index folder]
if __name__ == "__main__":
    import time

    # index folder from the command line, default "output"
    folder = sys.argv[1] if len(sys.argv) > 1 else "output"
    # start the load timer
    t0 = time.time()
    # load the index and close it automatically
    with IndexReader(folder) as idx:
        # load statistics
        print(f"loaded in {time.time() - t0:.2f}s: N={idx.N:,}, vocab={idx.vocab_size:,}, log_base={idx.log_base}")
        # sample terms to show df / idf and the first postings
        for t in ("hong", "kong", "busi", "microsoft"):
            # first three (doc_id, tf) pairs of the term
            head = idx.get_doc_tfs(t)[:3]
            # one summary line per term
            print(f"{t:10s} df={idx.df(t):>6,} idf={idx.idf(t):.4f} first={head}")
        # prefix search demo
        print("prefix 'micros':", idx.terms_with_prefix("micros")[:10])
        # TREC-ID lookup demo
        print("doc 1 ->", idx.doc_label(1), idx.urls[1])
