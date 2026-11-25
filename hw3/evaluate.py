"""Independent correctness verification and inexact top-k performance benchmark suite.

Validates search.py's scoring and ranking accuracy against a standalone brute-force
reference implementation across 11 weighting schemes (SMART variants and Okapi BM25),
and measures per-query latency, postings scored, and top-10 overlap across 14 exact
and inexact retrieval configurations on 200 TREC Web Track (2009-2012) topics.
"""

import argparse
import array
import math
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

# index access and HW1's parser
from index_reader import IndexReader, hw1
# query parsing
from query_parser import parse_query
# ranking under test
from ranker import Ranker
# batch query file reader of the CLI
from search import iter_batch
# weighting schemes
from weighting import Scheme

# text shown by --help
HELP_TEXT = """Checks and measurements for the documentation (not needed for searching).

  python evaluate.py correctness     search.py's scores against an independent brute-force scorer
  python evaluate.py speed           time, postings scored and top-10 overlap of every inexact top-k method
  python evaluate.py all             both; writes results/evaluation.md"""
# folder of this script
_HERE = os.path.dirname(os.path.abspath(__file__))
# schemes checked against the brute-force scorer
CHECK_SCHEMES = ["ltc.ltc", "lnc.ltc", "nnc.ntc", "ntn.ntn", "anc.atc", "atc.atc", "bnn.bpn", "Lpc.ltc", "Ltc.Ltc",
                 "lpc.apc", "bm25"]
# hand-written cases next to the slide examples and the Web Track topics
EDGE_QUERIES = ["Microsoft", "airplanes", "United States government", "non-profit organization", "Hong Kong business",
                "open source AND Linux", "software engineer OR project manager", "the of and", "to be or not to be",
                "zzzqqqxxy", "zzzqqqxxy linux", "state-of-the-art", "3.5mm jack", "403b", "windows 7 and office 2007",
                "new york new york", "information information retrieval", "home page", "free", "HONG kong",
                "linux AND zzzqqqxxy", "linux OR zzzqqqxxy", "the AND linux", "a OR b"]


# ----------------------------------------------------------------------
# brute-force reference: nothing is shared with search.py except HW1's tokenizer and record parser
# ----------------------------------------------------------------------

# The whole index as plain per-document vectors, read from postings.txt with HW1's own parser.
# independent reference scorer
class BruteForce:
    """Standalone reference scorer built directly from postings.txt.

    Shares only HW1's tokenizer and line parser with search.py. Constructs full
    per-document term-frequency and document-frequency vectors in memory and
    evaluates SMART (ddd.qqq) and BM25 definitions from scratch.

    Attributes:
        n (int): Total number of documents in the collection (N = 31,480).
        base (float): Logarithm base read from meta.json.
        avgdl (float): Average document length across the collection.
        postings (Dict[str, Dict[int, int]]): Mapping from term to {doc_id: tf}.
        doc_tf (Dict[int, array.array]): Per-document parallel uint32 array of term frequencies.
        doc_df (Dict[int, array.array]): Per-document parallel uint32 array of document frequencies.
    """

    # load every posting of postings.txt into plain dictionaries
    def __init__(self, index_dir: str, n: int, log_base: float, avgdl: float):
        # collection size, log base and average document length
        self.n, self.base, self.avgdl = n, log_base, avgdl
        # term -> {doc_id: tf}
        self.postings: Dict[str, Dict[int, int]] = {}
        # doc_id -> tf of every term, and df of the same terms (parallel arrays)
        self.doc_tf: Dict[int, "array.array"] = {}
        # df values parallel to doc_tf
        self.doc_df: Dict[int, "array.array"] = {}
        # (document letters, doc_id) -> vector length, so a document's vector is built once per scheme
        self._norms: Dict[Tuple[str, int], float] = {}
        # full pass over postings.txt in binary mode
        with open(os.path.join(index_dir, "postings.txt"), "rb") as fp:
            # one record per line
            for raw in fp:
                # skip blank lines
                if not raw.strip():
                    # next record
                    continue
                # HW1's own parser
                term, df, plist = hw1.parse_posting_line(raw.decode("utf-8"), attach_skips=False)
                # term -> {doc_id: tf}
                self.postings[term] = {p.doc_id: p.tf for p in plist}
                # every posting of the term
                for p in plist:
                    # first term seen for this document
                    if p.doc_id not in self.doc_tf:
                        # new tf array
                        self.doc_tf[p.doc_id] = array.array("I")
                        # new df array
                        self.doc_df[p.doc_id] = array.array("I")
                    # tf of this term in the document
                    self.doc_tf[p.doc_id].append(p.tf)
                    # df of this term
                    self.doc_df[p.doc_id].append(df)

    # logarithm in the index's base
    def log(self, x: float) -> float:
        # same base as meta.json
        return math.log(x, self.base)

    # SMART tf component
    def tf_part(self, letter: str, tf: int, max_tf: int, avg_tf: float) -> float:
        # natural
        if letter == "n":
            # raw tf
            return float(tf)
        # logarithmic
        if letter == "l":
            # 1 + log tf
            return 1 + self.log(tf)
        # boolean
        if letter == "b":
            # constant
            return 1.0
        # augmented
        if letter == "a":
            # 0.5 + 0.5 tf / max tf
            return 0.5 + 0.5 * tf / max_tf
        # "L": (1 + log tf) / (1 + log avg tf)
        return (1 + self.log(tf)) / (1 + self.log(avg_tf))

    # SMART idf component
    def idf_part(self, letter: str, df: int) -> float:
        # no idf
        if letter == "n":
            # constant
            return 1.0
        # log(N / df)
        if letter == "t":
            # standard idf
            return self.log(self.n / df)
        # "p": probabilistic idf, clipped at 0
        return max(0.0, self.log((self.n - df) / df)) if df < self.n else 0.0

    # full weight vector of one side: [(term-or-index, weight)] with cosine normalization when asked
    def weights(self, letters: str, tf_df: List[Tuple[int, int]]) -> List[float]:
        # tf values
        tfs = [tf for tf, _ in tf_df]
        # max and average tf of this vector
        max_tf, avg_tf = max(tfs), sum(tfs) / len(tfs)
        # tf part times idf part of every component
        w = [self.tf_part(letters[0], tf, max_tf, avg_tf) * self.idf_part(letters[1], df) for tf, df in tf_df]
        # cosine normalization
        if letters[2] == "c":
            # vector length
            norm = math.sqrt(sum(v * v for v in w))
            # divide every component (0 for a zero vector)
            w = [v / norm if norm > 0 else 0.0 for v in w]
        # weights in input order
        return w

    # query string -> (term -> query tf, set of documents allowed by AND / OR, or None for free text)
    def understand(self, text: str, cfg) -> Tuple[Counter, Optional[Set[int]]]:
        # processed terms of a string, with HW1's tokenizer and linguistic module
        terms_of = lambda s: [t for t, _ in hw1.linguistic(hw1.tokenize(s, cfg), cfg)]
        # raw words
        words = text.split()
        # term -> query tf
        qtf: Counter = Counter()
        # no Boolean operator: free text
        if "AND" not in words and "OR" not in words:
            # count every processed term
            qtf.update(terms_of(text))
            # no filter
            return qtf, None
        # OR of groups; inside a group every word must occur (explicit AND or adjacency)
        allowed: Set[int] = set()
        # every OR alternative
        for group in " ".join(words).split(" OR "):
            # documents containing every word of the group so far (None = no word yet)
            docs: Optional[Set[int]] = None
            # every word of the group
            for word in group.split():
                # explicit AND is the same as adjacency
                if word == "AND":
                    # skip the operator
                    continue
                # a word that becomes several terms ("non-profit") must occur as that phrase: not used in these tests
                ts = terms_of(word)
                # at most one term per word in Boolean test queries
                assert len(ts) <= 1, f"brute force does not handle multi-term word {word!r} in Boolean queries"
                # the word's term (none for a stopword)
                for t in ts:
                    # count it for scoring
                    qtf[t] += 1
                    # documents containing it
                    have = set(self.postings.get(t, ()))
                    # intersect with the group so far
                    docs = have if docs is None else docs & have
            # group with at least one real term
            if docs is not None:
                # OR: union of the groups
                allowed |= docs
        # query tf and filter
        return qtf, allowed

    # every scored document: {doc_id: score}, straight from the definitions
    def score(self, text: str, cfg, scheme: str, k1: float = 1.2, b: float = 0.75) -> Dict[int, float]:
        # query tf and Boolean filter
        qtf, allowed = self.understand(text, cfg)
        # query terms that exist in the collection and are not in every document
        q_terms = [t for t in qtf if t in self.postings and len(self.postings[t]) < self.n]
        # no usable term
        if not q_terms:
            # Boolean matches score 0; free text has no result
            return {d: 0.0 for d in allowed} if allowed else {}
        # df of every query term
        dfs = {t: len(self.postings[t]) for t in q_terms}
        # documents containing at least one query term
        docs = set().union(*(self.postings[t].keys() for t in q_terms))
        # Boolean filter
        if allowed is not None:
            # keep matching documents only
            docs &= allowed
        # doc_id -> score
        out: Dict[int, float] = {}
        # BM25 by its formula
        if scheme == "bm25":
            # every candidate document
            for d in docs:
                # document length = sum of its tf values
                dl = sum(self.doc_tf[d])
                # score of this document
                s = 0.0
                # every query term
                for t in q_terms:
                    # tf of the term in the document
                    tf = self.postings[t].get(d, 0)
                    # term present
                    if tf:
                        # smoothed BM25 idf
                        idf = math.log(1 + (self.n - dfs[t] + 0.5) / (dfs[t] + 0.5))
                        # qtf * idf * tf part
                        s += qtf[t] * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                # store
                out[d] = s
        # SMART ddd.qqq
        else:
            # document letters and query letters
            d_letters, q_letters = scheme.split(".")
            # query weights by the definition
            wq = dict(zip(q_terms, self.weights(q_letters, [(qtf[t], dfs[t]) for t in q_terms])))
            # every candidate document
            for d in docs:
                # tf values of the whole document
                tfs = self.doc_tf[d]
                # its max and average tf
                max_tf, avg_tf = max(tfs), sum(tfs) / len(tfs)
                # vector length from the document's full vector, computed from scratch (kept per document and scheme)
                norm = 1.0
                # cosine normalization on the document side
                if d_letters[2] == "c":
                    # computed before for this scheme and document
                    norm = self._norms.get((d_letters, d))
                    # first time
                    if norm is None:
                        # un-normalized weight of every term of the document
                        raw = [self.tf_part(d_letters[0], tf, max_tf, avg_tf) * self.idf_part(d_letters[1], df)
                               for tf, df in zip(tfs, self.doc_df[d])]
                        # vector length, cached per scheme and document
                        norm = self._norms[(d_letters, d)] = math.sqrt(sum(v * v for v in raw))
                # score of this document
                s = 0.0
                # every query term
                for t in q_terms:
                    # tf of the term in the document
                    tf = self.postings[t].get(d, 0)
                    # term present
                    if tf:
                        # document weight by the definition
                        wd = self.tf_part(d_letters[0], tf, max_tf, avg_tf) * self.idf_part(d_letters[1], dfs[t])
                        # add the normalized product
                        s += wq[t] * wd / norm if norm > 0 else 0.0
                # zero-length documents cannot be ranked
                if norm > 0:
                    # store
                    out[d] = s
            # a query vector of length 0 ranks nothing
            if q_letters[2] == "c" and not any(wq.values()):
                # nothing ranked
                out = {}
        # Boolean matches that scored nothing still belong to the result, with score 0
        if allowed is not None:
            # every matching document
            for d in allowed:
                # 0 unless it scored
                out.setdefault(d, 0.0)
        # doc_id -> score
        return out


# compare search.py with the brute-force scorer; returns the report lines
def check_correctness(index_dir: str, queries: List[Tuple[str, str]], schemes: List[str], scheme_queries: int) -> List[str]:
    # markdown report
    lines = ["## Correctness: search.py against an independent brute-force scorer", ""]
    # open the index under test
    with IndexReader(index_dir) as idx:
        # start timer
        t0 = time.time()
        # load the reference scorer
        brute = BruteForce(index_dir, idx.N, idx.log_base, idx.meta["avgdl"])
        # progress on stderr
        print(f"brute-force index loaded with HW1's parser ({time.time() - t0:.0f}s)", file=sys.stderr)
        # table header
        lines += ["| scheme | queries | documents compared | max abs. score difference | rankings identical |",
                  "|---|---|---|---|---|"]
        # every scheme to check
        for scheme in schemes:
            # ranker under test
            ranker = Ranker(idx, scheme=Scheme(idx, scheme))
            # every query for the Slide 4 scheme, a fixed subset for the others
            subset = queries if scheme == "ltc.ltc" else queries[:scheme_queries]
            # worst difference, documents compared, identical rankings, empty results
            worst, compared, same_rank, empty = 0.0, 0, 0, 0
            # every query of the subset
            for qid, text in subset:
                # every scored document from search.py's ranker
                got = dict(ranker.rank(parse_query(text, idx.cfg), top_k=0).hits)
                # the same from the reference scorer
                want = brute.score(text, idx.cfg, scheme)
                # same documents
                assert got.keys() == want.keys(), f"{scheme} {qid} {text!r}: different documents " \
                                                  f"({len(got)} vs {len(want)})"
                # largest score difference of this query
                diff = max((abs(got[d] - want[d]) for d in got), default=0.0)
                # same scores up to rounding
                assert diff < 1e-9, f"{scheme} {qid} {text!r}: score difference {diff}"
                # track the worst difference
                worst = max(worst, diff)
                # documents compared
                compared += len(got)
                # queries without a result
                empty += not got
                # ranking: score descending, doc_id ascending; scores closer than 1e-12 count as tied
                order = lambda s: [d for d, _ in sorted(s.items(), key=lambda x: (-round(x[1], 12), x[0]))]
                # same top-100 order
                same_rank += order(got)[:100] == order(want)[:100]
            # one table row per scheme
            lines.append(f"| {scheme} | {len(subset)} ({empty} with no result) | {compared:,} | {worst:.1e} | "
                         f"{same_rank} / {len(subset)} |")
            # progress on stderr
            print(lines[-1], file=sys.stderr)
    # explanation under the table
    lines += ["", "Every score of every matching document is compared (top-k = all). The brute-force scorer shares only "
              "HW1's tokenizer and record parser with search.py: it builds full document vectors from postings.txt and "
              "applies the SMART / BM25 definitions directly, including each document's vector length.", ""]
    # report lines
    return lines


# ----------------------------------------------------------------------
# speed and overlap of the inexact top-k methods
# ----------------------------------------------------------------------

# (label, scheme, rank() options, label of the exact run it is compared with)
METHODS = [
    ("exact, postings.txt (baseline)", "ltc.ltc", {}, None),
    ("exact, tf index (`--tf-index`)", "ltc.ltc", {"tf_index": True}, 0),
    ("index elimination `--idf-threshold 1.0`", "ltc.ltc", {"idf_threshold": 1.0}, 0),
    ("index elimination `--min-match 2`", "ltc.ltc", {"min_match": 2}, 0),
    ("champion lists `--champions` (r = 50)", "ltc.ltc", {"champions": True}, 0),
    ("tiered index `--tiered`", "ltc.ltc", {"tiered": True}, 0),
    ("impact ordering `--impact`", "ltc.ltc", {"impact": True}, 0),
    ("cluster pruning `--cluster 1,4`", "ltc.ltc", {"cluster": (1, 4)}, 0),
    ("cluster pruning `--cluster 3,8`", "ltc.ltc", {"cluster": (3, 8)}, 0),
    ("impact + cluster `--impact --cluster 3,8`", "ltc.ltc", {"impact": True, "cluster": (3, 8)}, 0),
    ("BM25 exact, postings.txt", "bm25", {}, None),
    ("BM25 `--champions`", "bm25", {"champions": True}, 10),
    ("BM25 `--tiered`", "bm25", {"tiered": True}, 10),
    ("BM25 `--impact`", "bm25", {"impact": True}, 10),
]


# run every method over the queries; returns the report lines
def measure_speed(index_dir: str, queries: List[Tuple[str, str]], repeats: int, k: int = 10) -> List[str]:
    # markdown report
    lines = [f"## Inexact top-{k}: time and agreement with the exact ranking", ""]
    # open the index
    with IndexReader(index_dir) as idx:
        # one ranker per scheme used by the methods
        rankers = {s: Ranker(idx, scheme=Scheme(idx, s)) for s in {m[1] for m in METHODS}}
        # parse every query once
        parsed = [parse_query(text, idx.cfg) for _, text in queries]
        # top-k doc_ids of every method, in METHODS order
        tops: List[List[List[int]]] = []
        # one table row per method
        rows = []
        # every method
        for label, scheme, opts, ref in METHODS:
            # ranker of its scheme
            ranker = rankers[scheme]
            # one untimed pass: fills the OS file cache and opens lazily loaded structures
            for q in parsed:
                # warm-up run
                ranker.rank(q, top_k=k, **opts)
            # best total time over the timed passes
            best = float("inf")
            # timed passes
            for _ in range(repeats):
                # start timer
                t0 = time.perf_counter()
                # every query
                results = [ranker.rank(q, top_k=k, **opts) for q in parsed]
                # keep the fastest pass
                best = min(best, time.perf_counter() - t0)
            # doc_ids of every result list
            top = [[d for d, _ in r.hits] for r in results]
            # remember for the methods compared with this one
            tops.append(top)
            # average postings scored per query
            scored = sum(r.postings_scored for r in results) / len(results)
            # queries whose search had to be widened
            widened = sum(r.champions_fallback for r in results)
            # exact runs have nothing to compare with
            if ref is None:
                # empty cells
                overlap, same = "-", "-"
            # inexact runs are compared with their exact reference
            else:
                # (method set, exact set) for queries where the exact run found something
                pairs = [(set(a), set(b)) for a, b in zip(top, tops[ref]) if b]
                # mean share of the exact top-k that the method also returns
                overlap = f"{sum(len(a & b) / len(b) for a, b in pairs) / len(pairs):.3f}"
                # identical lists, same order
                same = f"{sum(a == b for a, b in zip(top, tops[ref]))} / {len(top)}"
            # one row: label, ms per query, postings, overlap, identical, widened
            rows.append((label, best / len(parsed) * 1000, scored, overlap, same, widened))
            # progress on stderr
            print(rows[-1], file=sys.stderr)
        # baseline time of each exact reference (row 0: ltc, row 10: BM25)
        base = {0: rows[0][1], 10: rows[10][1]}
        # table header
        lines += ["| method | ms / query | speed-up | postings scored / query | mean overlap@10 | identical top-10 "
                  "(same order) | queries widened |", "|---|---|---|---|---|---|---|"]
        # every row with its reference
        for (label, ms, scored, overlap, same, widened), (_, _, _, ref) in zip(rows, METHODS):
            # speed-up relative to the exact reference
            speed = "1.00x" if ref is None else f"{base[ref] / ms:.2f}x"
            # table row
            lines.append(f"| {label} | {ms:.2f} | {speed} | {scored:,.0f} | {overlap} | {same} | {widened} |")
    # explanation under the table
    lines += ["", f"{len(queries)} free-text queries (TREC Web Track 2009-2012 topics), top-{k}, best of {repeats} "
              "timed passes after one warm-up pass, single process, postings in the OS file cache. Overlap is the "
              "share of the exact top-10 that the method also returns, averaged over the queries with at least one "
              "result. \"Queries widened\" counts queries where the pruned candidate set held fewer than 10 documents "
              "and the search fell back to a wider or exact set (that time is included).", ""]
    # report lines
    return lines


# entry point
def main(argv: Optional[List[str]] = None) -> int:
    # keep the help text's line breaks
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # what to run
    ap.add_argument("what", choices=("correctness", "speed", "all"))
    # index folder
    ap.add_argument("-i", "--index", default=os.path.join(_HERE, "output"))
    # topic file
    ap.add_argument("-f", "--file", default=os.path.join(_HERE, "queries_webtrack.txt"), help="query file")
    # report file (written by "all" only)
    ap.add_argument("-o", "--output", default=os.path.join(_HERE, "results", "evaluation.md"))
    # size of the query subset for the non-default schemes
    ap.add_argument("--scheme-queries", type=int, default=60, help="queries per non-default scheme (default: 60)")
    # timed passes
    ap.add_argument("--repeats", type=int, default=3, help="timed passes per method (default: 3)")
    # parse the command line
    args = ap.parse_args(argv)
    # never crash on characters the console cannot show
    sys.stderr.reconfigure(errors="replace")
    # (qid, query) pairs of the topic file
    topics = list(iter_batch(args.file))
    # report title
    report = ["# Evaluation (generated by `python evaluate.py all`)", ""]
    # correctness part
    if args.what in ("correctness", "all"):
        # edge cases first, so they are part of every scheme's subset
        cases = [(f"e{i}", q) for i, q in enumerate(EDGE_QUERIES, 1)] + topics
        # compare with the brute-force scorer
        report += check_correctness(args.index, cases, CHECK_SCHEMES, args.scheme_queries)
    # speed part
    if args.what in ("speed", "all"):
        # time every method
        report += measure_speed(args.index, topics, args.repeats)
    # whole report
    text = "\n".join(report)
    # "all" also writes the report file
    if args.what == "all":
        # LF-only markdown
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            # write it
            f.write(text)
    # and prints it
    print(text)
    # success
    return 0


# run as a script
if __name__ == "__main__":
    # exit with main's code
    sys.exit(main())
