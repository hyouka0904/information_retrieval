"""SMART weighting scheme (ddd.qqq) and Okapi BM25 scoring engine.

900 combinations of SMART notation on both document and query, alongside parameterized Okapi BM25:
5 term-frequency variants ('n', 'l', 'a', 'b', 'L')
x 3 inverse document frequency variants ('n', 't', 'p')
x 2 normalization modes ('n', 'c')
Lazily loads precomputed per-document statistics and vector norms from doc_stats.tsv when required by the active scheme.
"""

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

# SMART letters that are implemented (IIR Fig. 6.15)
TF_LETTERS = "nlabL"
# idf letters: none, log(N/df), probabilistic
IDF_LETTERS = "ntp"
# normalization letters: none, cosine
NORM_LETTERS = "nc"
# Slide 4 formula on both sides
DEFAULT_SCHEME = "ltc.ltc"
# BM25 defaults (the values BM25 champion lists were built with)
DEFAULT_K1 = 1.2
# BM25 length normalization default
DEFAULT_B = 0.75
# per-document statistics and vector lengths written by prep_extra.py
DOC_STATS_FILE = "doc_stats.tsv"

# text shown when a scheme name cannot be parsed
HELP = """weighting scheme: SMART notation ddd.qqq (document.query) or bm25.
  tf   n natural   l 1+log tf   a 0.5+0.5 tf/max tf   b boolean   L (1+log tf)/(1+log avg tf)
  idf  n none      t log(N/df)  p max(0, log((N-df)/df))
  norm n none      c cosine
  examples: ltc.ltc (default, Slide 4), lnc.ltc (textbook), nnc.ntc, anc.atc, Lpc.ltc, bm25"""


# raised for a scheme name that cannot be understood
class SchemeError(ValueError):
    # no behaviour beyond ValueError; the type lets callers catch scheme errors only
    pass


# document weight, query weight and normalization of one scheme
class Scheme:
    """Encapsulates document weighting, query weighting, and vector normalization for a scheme.

    Computes score(q, d) = sum_t wq(t) * wd(t, d) / (query_norm * doc_norm(d)).
    With 'ltc.ltc', reuses IndexReader.doc_norms directly without loading doc_stats.tsv.

    Attributes:
        idx (IndexReader): Underlying index reader.
        spec (str): Canonical scheme specification ('ddd.qqq' or 'bm25').
        is_bm25 (bool): True if the scheme is Okapi BM25.
        k1 (float): BM25 tf saturation constant (default 1.2).
        b (float): BM25 document length normalization constant (default 0.75).
    """

    # parse the scheme name; SchemeError for anything that is not ddd.qqq, ddd or bm25
    def __init__(self, idx, spec: str = DEFAULT_SCHEME, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        # index access (N, df, log base)
        self.idx = idx
        # name as typed
        self.spec = spec
        # BM25 is named as a whole, case-insensitively
        self.is_bm25 = spec.lower() == "bm25"
        # BM25 parameters (unused by SMART schemes)
        self.k1, self.b = k1, b
        # BM25
        if self.is_bm25:
            # canonical name
            self.spec = "bm25"
            # BM25 has no vector normalization on either side
            self.doc, self.query = "bm25", "bm25"
        # SMART notation
        else:
            # document part and query part
            parts = spec.split(".")
            # "lnc" alone means the same letters on both sides
            if len(parts) == 1:
                # duplicate it
                parts = parts * 2
            # exactly two parts of three letters each
            if len(parts) != 2 or any(len(p) != 3 for p in parts):
                # reject the shape
                raise SchemeError(f"scheme {spec!r} is not of the form ddd.qqq or bm25")
            # check every letter of both parts
            for p in parts:
                # tf, idf and normalization letter must all be known
                if p[0] not in TF_LETTERS or p[1] not in IDF_LETTERS or p[2] not in NORM_LETTERS:
                    # name the allowed letters
                    raise SchemeError(f"scheme {spec!r}: tf must be one of {TF_LETTERS}, idf one of {IDF_LETTERS}, "
                                      f"normalization one of {NORM_LETTERS}")
            # document letters and query letters
            self.doc, self.query = parts
            # canonical name ("lnc" -> "lnc.lnc")
            self.spec = ".".join(parts)
        # True for the Slide 4 formula
        self.is_default = self.spec == DEFAULT_SCHEME
        # per-document tables, loaded only when a scheme needs them: max tf
        self._max_tf: Optional[List[int]] = None
        # average tf per document
        self._avg_tf: Optional[List[float]] = None
        # document vector lengths under this scheme
        self._norms: Optional[List[float]] = None
        # BM25 denominator constants
        self._bm_k: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # per-document tables
    # ------------------------------------------------------------------

    # read the wanted columns of doc_stats.tsv: column name -> list indexed by doc_id
    def _load_stats(self, columns: List[str]) -> Dict[str, List[float]]:
        # file written by prep_extra.py
        path = os.path.join(self.idx.index_dir, DOC_STATS_FILE)
        # never built
        if not os.path.exists(path):
            # tell the user how to build it
            raise FileNotFoundError(f"{path} not found; build it with 'python prep_extra.py' first")
        # one zero-filled list per column, indexed by doc_id
        out = {c: [0.0] * (self.idx.N + 1) for c in columns}
        # LF-only line splitting, matching how the file was written
        with open(path, encoding="utf-8", newline="\n") as f:
            # column names
            header = f.readline().rstrip("\n").split("\t")
            # column name -> field index
            where = {c: header.index(c) for c in columns}
            # one document per line
            for line in f:
                # fields of this document
                cells = line.rstrip("\n").split("\t")
                # first field is the doc_id
                d = int(cells[0])
                # copy every wanted column
                for c, k in where.items():
                    # store as float
                    out[c][d] = float(cells[k])
        # column name -> values
        return out

    # max tf of every document (augmented tf)
    def _doc_max_tf(self) -> List[float]:
        # load on first use
        if self._max_tf is None:
            # one column of doc_stats.tsv
            self._max_tf = self._load_stats(["max_tf"])["max_tf"]
        # cached list
        return self._max_tf

    # average tf of every document = |d| / number of distinct terms (log-average tf)
    def _doc_avg_tf(self) -> List[float]:
        # load on first use
        if self._avg_tf is None:
            # document length and number of distinct terms
            t = self._load_stats(["length", "distinct"])
            # their ratio; 0 for documents without terms
            self._avg_tf = [(l / u if u else 0.0) for l, u in zip(t["length"], t["distinct"])]
        # cached list
        return self._avg_tf

    # ||d|| of every document under the document side of this scheme (all 1 when it is not cosine-normalized)
    def doc_norms(self) -> List[float]:
        # compute on first use
        if self._norms is None:
            # no normalization: dividing by 1 changes nothing
            if self.is_bm25 or self.doc[2] == "n":
                # all ones
                self._norms = [1.0] * (self.idx.N + 1)
            # Slide 4 weights: the table the index reader already holds
            elif self.doc[:2] == "lt":
                # reuse doc_norms.tsv
                self._norms = self.idx.doc_norms
            # any other tf/idf pair: its precomputed column ("norm_" + the two letters)
            else:
                # one column of doc_stats.tsv
                self._norms = self._load_stats(["norm_" + self.doc[:2]])["norm_" + self.doc[:2]]
        # cached list
        return self._norms

    # BM25 denominator part k1 (1 - b + b |d| / avgdl) of every document
    def bm25_k(self) -> List[float]:
        # compute on first use
        if self._bm_k is None:
            # average document length written by prep.py
            avgdl = self.idx.meta["avgdl"]
            # documents missing from the file have length 0
            k = [self.k1 * (1.0 - self.b)] * (self.idx.N + 1)
            # document lengths written by prep.py
            with open(os.path.join(self.idx.index_dir, "doc_lengths.tsv"), encoding="utf-8", newline="\n") as f:
                # one "doc_id<TAB>length" per line
                for line in f:
                    # the two fields
                    d_s, l_s = line.rstrip("\n").split("\t")
                    # length-normalized saturation constant of this document
                    k[int(d_s)] = self.k1 * (1.0 - self.b + self.b * int(l_s) / avgdl)
            # cache
            self._bm_k = k
        # cached list
        return self._bm_k

    # ------------------------------------------------------------------
    # idf
    # ------------------------------------------------------------------

    # idf of a processed term under one SMART idf letter
    def idf(self, term: str, letter: str) -> float:
        # document frequency of the term
        df = self.idx.df(term)
        # a term absent from the index has no weight under any idf
        if df == 0:
            # no weight
            return 0.0
        # "n": no idf
        if letter == "n":
            # constant weight
            return 1.0
        # "t": log(N / df), the index reader's own idf
        if letter == "t":
            # same value as the original ranker
            return self.idx.idf(term)
        # probabilistic idf, clipped at 0 for terms in more than half of the documents
        n = self.idx.N
        # log((N - df) / df), 0 for df >= N
        return max(0.0, math.log((n - df) / df, self.idx.log_base)) if df < n else 0.0

    # BM25 idf: ln(1 + (N - df + 0.5) / (df + 0.5)), always positive
    def bm25_idf(self, term: str) -> float:
        # document frequency of the term
        df = self.idx.df(term)
        # 0 for a term absent from the index
        return math.log(1.0 + (self.idx.N - df + 0.5) / (df + 0.5)) if df else 0.0

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    # function (doc_id, tf) -> un-normalized document weight of one term
    def doc_weight_fn(self, term: str) -> Callable[[int, int], float]:
        # BM25
        if self.is_bm25:
            # term idf, k1 + 1 and the per-document denominator constants
            idf, k1p, bm_k = self.bm25_idf(term), self.k1 + 1.0, self.bm25_k()
            # idf * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
            return lambda d, tf: idf * (tf * k1p / (tf + bm_k[d]))
        # SMART tf letter of the document side
        letter = self.doc[0]
        # SMART idf of the document side
        idf = self.idf(term, self.doc[1])
        # 1 + log tf, as defined by the index reader
        tfw = self.idx.tf_weight
        # "l": logarithmic tf
        if letter == "l":
            # same expression as the original ranker: (1 + log tf) * idf
            return lambda d, tf: tfw(tf) * idf
        # "n": natural tf
        if letter == "n":
            # raw tf
            return lambda d, tf: tf * idf
        # "b": boolean tf
        if letter == "b":
            # 1 for every document containing the term
            return lambda d, tf: idf
        # "a": augmented tf
        if letter == "a":
            # per-document max tf
            max_tf = self._doc_max_tf()
            # 0.5 + 0.5 tf / max tf
            return lambda d, tf: (0.5 + 0.5 * tf / max_tf[d]) * idf
        # "L": log tf relative to the document's average tf
        avg_tf = self._doc_avg_tf()
        # (1 + log tf) / (1 + log avg tf)
        return lambda d, tf: tfw(tf) / tfw(avg_tf[d]) * idf

    # (term -> query weight, query vector length) for the query term frequencies
    def query_weights(self, qtf: Dict[str, int]) -> Tuple[Dict[str, float], float]:
        # BM25
        if self.is_bm25:
            # a repeated query word counts once per occurrence
            return {t: float(c) for t, c in qtf.items()}, 1.0
        # SMART tf letter of the query side
        letter = self.query[0]
        # 1 + log tf, as defined by the index reader
        tfw = self.idx.tf_weight
        # max query tf, for "a"
        max_q = max(qtf.values(), default=1)
        # average query tf, for "L"
        avg_q = sum(qtf.values()) / len(qtf) if qtf else 1.0
        # term -> query weight
        w: Dict[str, float] = {}
        # every query term
        for t, c in qtf.items():
            # SMART idf of the query side
            idf = self.idf(t, self.query[1])
            # "l": logarithmic tf
            if letter == "l":
                # (1 + log qtf) * idf
                w[t] = tfw(c) * idf
            # "n": natural tf
            elif letter == "n":
                # qtf * idf
                w[t] = c * idf
            # "b": boolean tf
            elif letter == "b":
                # idf only
                w[t] = idf
            # "a": augmented tf
            elif letter == "a":
                # (0.5 + 0.5 qtf / max qtf) * idf
                w[t] = (0.5 + 0.5 * c / max_q) * idf
            # "L": log tf relative to the average query tf
            else:
                # (1 + log qtf) / (1 + log avg qtf) * idf
                w[t] = tfw(c) / tfw(avg_q) * idf
        # query vector length for "c", 1 otherwise
        norm = math.sqrt(sum(v ** 2 for v in w.values())) if self.query[2] == "c" else 1.0
        # weights and length
        return w, norm

    # which precomputed max-score bound of tf_index.bin is valid for this scheme: "ltc", "bm25" or None
    def bound_kind(self, tf_index_meta: dict) -> Optional[str]:
        # BM25 bounds are valid only for the k1 / b they were computed with
        if self.is_bm25:
            # parameters the bounds were built with
            bm = tf_index_meta.get("bm25", {})
            # same parameters -> usable
            return "bm25" if (bm.get("k1"), bm.get("b")) == (self.k1, self.b) else None
        # the cosine bound assumes ltc weights on the document side
        return "ltc" if self.doc == "ltc" else None
