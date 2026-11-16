import heapq
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# index access layer
from index_reader import IndexReader
# Boolean / positional filter evaluation
from matcher import Matcher
# AST node types and the helper that lists ranking leaves
from query_parser import Phrase, Proximity, Query, Term, Wildcard, positive_leaves

# weighting schemes the ranker can score with
SCHEMES = ("tfidf", "bm25")
# champion scheme name in meta.json for each weighting scheme
CHAMPION_SCHEME_OF = {"tfidf": "cosine", "bm25": "bm25"}
# champion file used when meta.json has no champion section (older prep output)
DEFAULT_CHAMPION_FILES = {"cosine": "champions.tsv", "bm25": "champions_bm25.tsv"}
# BM25 tf saturation parameter used when neither the caller nor meta.json gives one
DEFAULT_BM25_K1 = 1.2
# BM25 length normalization parameter used when neither the caller nor meta.json gives one
DEFAULT_BM25_B = 0.75
# tiered index file used when meta.json does not name one
DEFAULT_TIER_FILES = {"cosine": "tiers.tsv", "bm25": "tiers_bm25.tsv"}


# result of Ranker.rank()
@dataclass
class RankResult:
    # (doc_id, score) pairs, best first
    hits: List[tuple]
    # number of documents that satisfied the filter / candidate rule
    num_candidates: int
    # processed query term -> query term frequency, after elimination
    terms_used: Dict[str, int]
    # query terms dropped by index elimination or missing from the index
    terms_dropped: List[str] = field(default_factory=list)
    # True when champion lists gave fewer than top_k documents and the full candidate set was used instead
    champions_fallback: bool = False
    # tiered index: how many tiers were read before top_k documents were found (None = tiers not used)
    tiers_used: Optional[int] = None
    # tiered index: total number of tiers, the bottom one being "all postings" (None = tiers not used)
    tiers_total: Optional[int] = None


# TF-IDF + cosine or BM25 ranking over an IndexReader, with a Matcher for the query filter
class Ranker:

    # wildcard_limit caps how many expansions of one wildcard take part in scoring; k1 / b default to meta.json
    def __init__(self, idx: IndexReader, matcher: Optional[Matcher] = None, wildcard_limit: int = 30,
                 scheme: str = "tfidf", k1: Optional[float] = None, b: Optional[float] = None):
        # only the listed weighting schemes exist
        if scheme not in SCHEMES:
            # reject typos early
            raise ValueError(f"unknown weighting scheme {scheme!r}; choose one of {SCHEMES}")
        # index access
        self.idx = idx
        # "tfidf" (Slide 4, cosine) or "bm25"
        self.scheme = scheme
        # BM25 parameters the champion lists were built with (empty when prep.py did not build them)
        built = idx.meta.get("champions", {}).get("schemes", {}).get("bm25", {})
        # k1: caller, then meta.json, then the textbook default
        self.k1 = float(k1 if k1 is not None else built.get("k1", DEFAULT_BM25_K1))
        # b: caller, then meta.json, then the textbook default
        self.b = float(b if b is not None else built.get("b", DEFAULT_BM25_B))
        # doc_id -> k1 * (1 - b + b * |d| / avgdl), built on the first BM25 query
        self._bm25_k: Optional[List[float]] = None
        # Boolean / positional filter evaluation (shares the wildcard cache)
        self.matcher = matcher if matcher is not None else Matcher(idx)
        # a wildcard adds at most this many expanded terms to the score (the most frequent ones)
        self.wildcard_limit = wildcard_limit
        # term -> raw "doc doc ..." text of its champion list, loaded on first use
        self._champions: Optional[Dict[str, str]] = None
        # term -> parsed champion doc_ids, filled only for terms that were actually queried
        self._champion_sets: Dict[str, Set[int]] = {}
        # term -> raw "tier-1 docs<TAB>tier-2 docs ..." text of the tiers file, loaded on first use
        self._tiers: Optional[Dict[str, str]] = None
        # (term, level) -> doc_ids in tiers 0..level of the term, filled only for queried terms
        self._tier_sets: Dict[tuple, Set[int]] = {}

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    # document-side weight (1 + log tf) * idf
    def doc_weight(self, tf: int, idf: float) -> float:
        # Slide 4 formula, document side
        return self.idx.tf_weight(tf) * idf

    # query-side weight, same formula as the document side
    def query_weight(self, qtf: int, idf: float) -> float:
        # the same formula on the query side
        return self.idx.tf_weight(qtf) * idf

    # BM25 idf ln(1 + (N - df + 0.5) / (df + 0.5)); never negative, unlike the plain Robertson-Sparck Jones form
    def bm25_idf(self, term: str) -> float:
        # document frequency of the term
        df = self.idx.df(term)
        # a term absent from the index contributes nothing
        if df == 0:
            # no weight
            return 0.0
        # Lucene-style smoothed idf
        return math.log(1.0 + (self.idx.N - df + 0.5) / (df + 0.5))

    # doc_id -> k1 * (1 - b + b * |d| / avgdl), read from doc_lengths.tsv once per process
    def bm25_denominators(self) -> List[float]:
        # already built
        if self._bm25_k is not None:
            # reuse
            return self._bm25_k
        # average document length written by prep.py
        avgdl = self.idx.meta.get("avgdl")
        # document lengths written by prep.py
        path = os.path.join(self.idx.index_dir, "doc_lengths.tsv")
        # BM25 statistics were never built
        if not avgdl or not os.path.exists(path):
            # tell the user how to create them
            raise FileNotFoundError(f"{path} or meta.json 'avgdl' missing; build them with 'python prep.py' first")
        # documents missing from the file have length 0, i.e. k1 * (1 - b)
        table = [self.k1 * (1.0 - self.b)] * (self.idx.N + 1)
        # LF-only line splitting, matching how the file was written
        with open(path, encoding="utf-8", newline="\n") as f:
            # one "doc_id<TAB>length" per line
            for line in f:
                # split into the two fields
                d_s, len_s = line.rstrip("\n").split("\t")
                # length-normalized saturation constant of this document
                table[int(d_s)] = self.k1 * (1.0 - self.b + self.b * int(len_s) / avgdl)
        # cache for later queries
        self._bm25_k = table
        # finished table
        return table

    # True when the BM25 champion lists ("champions") or tiers ("tiers") were built with other k1 / b than scoring uses
    def champion_params_differ(self, section: str = "champions") -> bool:
        # only BM25 orderings depend on k1 / b
        if self.scheme != "bm25":
            # cosine orderings have no parameters
            return False
        # parameters recorded by prep.py
        built = (self.idx.meta.get(section) or {}).get("schemes", {}).get("bm25")
        # nothing recorded: cannot tell, so do not warn
        if not built:
            # no warning
            return False
        # compare with a tolerance for float round-off
        return abs(built["k1"] - self.k1) > 1e-9 or abs(built["b"] - self.b) > 1e-9

    # ------------------------------------------------------------------
    # query terms
    # ------------------------------------------------------------------

    # processed term -> its frequency in the query, from every leaf not under a NOT
    def query_terms(self, query: Query) -> Counter:
        # term -> query tf
        qtf: Counter = Counter()
        # every node that contributes to ranking
        for root in query.scoring:
            # leaves outside NOT subtrees
            for leaf in positive_leaves(root):
                # a single term
                if isinstance(leaf, Term):
                    # count it once
                    qtf[leaf.term] += 1
                # every word of a phrase is a query term
                elif isinstance(leaf, Phrase):
                    # each phrase term
                    for t, _ in leaf.terms:
                        # count it
                        qtf[t] += 1
                # both sides of a proximity pair are query terms
                elif isinstance(leaf, Proximity):
                    # left operand
                    qtf[leaf.left] += 1
                    # right operand
                    qtf[leaf.right] += 1
                # a wildcard contributes its most frequent expansions
                elif isinstance(leaf, Wildcard):
                    # every index term matching the pattern
                    expanded = self.matcher.expand_wildcard(leaf.pattern)
                    # highest df first; a cap keeps "a*" from adding thousands of rare terms
                    for t in sorted(expanded, key=self.idx.df, reverse=True)[: self.wildcard_limit]:
                        # count it
                        qtf[t] += 1
        # term frequencies of the query
        return qtf

    # ------------------------------------------------------------------
    # champion lists
    # ------------------------------------------------------------------

    # term -> raw id list of the champion file of the current scheme, read once per process
    def _load_champions(self) -> Dict[str, str]:
        # already loaded
        if self._champions is not None:
            # reuse
            return self._champions
        # champion ordering that matches the weighting scheme
        champ_scheme = CHAMPION_SCHEME_OF[self.scheme]
        # file name recorded by prep.py, or the default name
        name = self.idx.meta.get("champions", {}).get("schemes", {}).get(champ_scheme, {}).get(
            "file", DEFAULT_CHAMPION_FILES[champ_scheme])
        # champion lists written by prep.py / the prep notebook
        path = os.path.join(self.idx.index_dir, name)
        # champion lists were never built
        if not os.path.exists(path):
            # tell the user how to create them
            raise FileNotFoundError(f"{path} not found; build it with 'python prep.py' first")
        # term -> unparsed id list; parsing all 1.8M ids up front would cost about a second per process
        champs: Dict[str, str] = {}
        # LF-only line splitting, matching how the file was written
        with open(path, encoding="utf-8", newline="\n") as f:
            # one "term<TAB>doc doc doc ..." per line
            for line in f:
                # split term from the id list
                term, ids = line.rstrip("\n").split("\t", 1)
                # keep the text, parsed later on demand
                champs[term] = ids
        # cache for later queries
        self._champions = champs
        # loaded table
        return champs

    # champion doc_ids of one processed term (empty for a term not in the index), parsed on first use
    def champion_docs(self, term: str) -> Set[int]:
        # parsed before
        if term not in self._champion_sets:
            # raw id list of the term
            ids = self._load_champions().get(term, "")
            # parse the space-separated doc_ids once
            self._champion_sets[term] = {int(x) for x in ids.split()}
        # cached set
        return self._champion_sets[term]

    # ------------------------------------------------------------------
    # tiered index
    # ------------------------------------------------------------------

    # number of stored tiers (one per rank boundary in meta.json); the implicit bottom tier is not counted
    def stored_tiers(self) -> int:
        # tiered index description written by prep.py
        info = self.idx.meta.get("tiers")
        # never built
        if not info:
            # tell the user how to create it
            raise FileNotFoundError("meta.json has no tiered index; build it with 'python prep.py' first")
        # one stored tier per rank boundary
        return len(info["sizes"])

    # term -> raw tier text of the tiers file of the current scheme, read once per process
    def _load_tiers(self) -> Dict[str, str]:
        # already loaded
        if self._tiers is not None:
            # reuse
            return self._tiers
        # tier ordering that matches the weighting scheme (the same as the champion ordering)
        tier_scheme = CHAMPION_SCHEME_OF[self.scheme]
        # file name recorded by prep.py, or the default name
        name = (self.idx.meta.get("tiers") or {}).get("schemes", {}).get(tier_scheme, {}).get(
            "file", DEFAULT_TIER_FILES[tier_scheme])
        # tiered index written by prep.py / the prep notebook
        path = os.path.join(self.idx.index_dir, name)
        # tiered index was never built
        if not os.path.exists(path):
            # tell the user how to create it
            raise FileNotFoundError(f"{path} not found; build it with 'python prep.py' first")
        # term -> unparsed tier fields
        tiers: Dict[str, str] = {}
        # LF-only line splitting, matching how the file was written
        with open(path, encoding="utf-8", newline="\n") as f:
            # one "term<TAB>tier-1 docs<TAB>tier-2 docs ..." per line
            for line in f:
                # split term from the tier fields
                term, rest = line.rstrip("\n").split("\t", 1)
                # keep the text, parsed later on demand
                tiers[term] = rest
        # cache for later queries
        self._tiers = tiers
        # loaded table
        return tiers

    # doc_ids of a processed term in stored tiers 0..level (cumulative, 0-based); a term absent from the file has none
    def tier_docs(self, term: str, level: int) -> Set[int]:
        # cache key
        key = (term, level)
        # parsed before
        if key not in self._tier_sets:
            # raw tier fields of the term ("" for a term not in the index)
            raw = self._load_tiers().get(term, "")
            # one field per stored tier
            fields = raw.split("\t") if raw else []
            # union of the fields up to and including this level
            self._tier_sets[key] = {int(x) for fld in fields[: level + 1] for x in fld.split()}
        # cached set
        return self._tier_sets[key]

    # ------------------------------------------------------------------
    # ranking
    # ------------------------------------------------------------------

    # rank documents for a parsed query; top_k <= 0 returns every candidate (see README for the options);
    # _tier_level is internal: the tier the tiered search has reached
    def rank(self, query: Query, top_k: int = 10, idf_threshold: Optional[float] = None,
             min_match: int = 1, champions: bool = False, tiered: bool = False,
             _tier_level: int = 0) -> RankResult:
        # both restrict the candidate set in different ways; combining them has no textbook meaning
        if champions and tiered:
            # reject the combination
            raise ValueError("champion lists and the tiered index cannot be used together")
        # sorted doc_ids from the filter, or None for plain free text
        filtered = self.matcher.match(query)
        # query term frequencies
        qtf = self.query_terms(query)

        # idf of every query term (0 for terms missing from the index)
        idf = {t: self.idx.idf(t) for t in qtf}
        # terms that do not occur in the collection cannot contribute
        dropped = [t for t in qtf if idf[t] == 0.0]
        # remaining terms
        kept = [t for t in qtf if idf[t] > 0.0]
        # BM25 scoring instead of cosine
        bm25 = self.scheme == "bm25"
        # query vector length over every term present in the index (before elimination, so scores stay
        # comparable whether or not elimination is switched on); BM25 does not normalize
        q_norm = 0.0 if bm25 else math.sqrt(sum(self.query_weight(qtf[t], idf[t]) ** 2 for t in kept))
        # doc_id -> k1 * (1 - b + b * |d| / avgdl), only needed for BM25
        bm25_k = self.bm25_denominators() if bm25 else None

        # index elimination by idf
        if idf_threshold is not None and kept:
            # terms rare enough to keep
            high = [t for t in kept if idf[t] >= idf_threshold]
            # never eliminate everything: keep at least the rarest term
            if not high:
                # the term with the largest idf
                high = [max(kept, key=lambda t: idf[t])]
            # report the eliminated terms
            dropped += [t for t in kept if t not in high]
            # continue with the surviving terms
            kept = high

        # restriction to champion / tier documents (None = no restriction)
        restrict: Optional[Set[int]] = None
        # champion lists requested
        if champions and kept:
            # union of the champion lists of the surviving terms
            restrict = set().union(*(self.champion_docs(t) for t in kept))
        # number of stored tiers (0 when the tiered index is not used)
        levels = self.stored_tiers() if tiered and kept else 0
        # tiered index requested and the search has not yet fallen through to the bottom tier
        if tiered and kept and _tier_level < levels:
            # union of tiers 0.._tier_level of the surviving terms
            restrict = set().union(*(self.tier_docs(t, _tier_level) for t in kept))

        # set version of the filter result for O(1) membership tests
        allowed: Optional[Set[int]] = set(filtered) if filtered is not None else None
        # champion / tier restriction applies on top of the filter
        if restrict is not None:
            # intersect with the filter, or use the restriction alone
            allowed = restrict if allowed is None else allowed & restrict

        # doc_id -> accumulated dot product
        acc: Dict[int, float] = {}
        # doc_id -> number of query terms it contains (for min_match)
        hits_per_doc: Counter = Counter()
        # BM25 saturation numerator factor
        k1p1 = self.k1 + 1.0
        # term-at-a-time accumulation
        for t in kept:
            # query-side weight: tf-idf for cosine, query tf times BM25 idf for BM25
            wq = qtf[t] * self.bm25_idf(t) if bm25 else self.query_weight(qtf[t], idf[t])
            # walk the term's postings
            for d, tf in self.idx.get_doc_tfs(t):
                # skip documents outside the filter / champion set
                if allowed is not None and d not in allowed:
                    # next posting
                    continue
                # BM25 term contribution
                if bm25:
                    # add qtf * idf * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
                    acc[d] = acc.get(d, 0.0) + wq * tf * k1p1 / (tf + bm25_k[d])
                # TF-IDF dot product
                else:
                    # add w(t, q) * w(t, d)
                    acc[d] = acc.get(d, 0.0) + wq * self.doc_weight(tf, idf[t])
                # one more query term found in this document
                hits_per_doc[d] += 1

        # min_match only applies to free text (Boolean results are already exact)
        if query.mode == "free" and min_match > 1 and kept:
            # cannot demand more terms than the query has
            need = min(min_match, len(kept))
            # keep documents containing enough query terms
            acc = {d: s for d, s in acc.items() if hits_per_doc[d] >= need}

        # doc_id -> final score
        scores: Dict[int, float] = {}
        # BM25 sums are already the final scores
        if bm25:
            # no length normalization afterwards
            scores = dict(acc)
        # cosine
        else:
            # normalize every accumulator
            for d, dot in acc.items():
                # document vector length
                d_norm = self.idx.doc_norms[d]
                # zero-length vectors cannot be normalized
                if d_norm > 0.0 and q_norm > 0.0:
                    # cosine similarity
                    scores[d] = dot / (q_norm * d_norm)

        # filtered queries keep matching documents that scored 0 (e.g. "NOT x", or docs matched only via OR NOT)
        if filtered is not None:
            # every filtered document that survived the champion / tier restriction
            for d in (filtered if restrict is None else (x for x in filtered if x in restrict)):
                # documents without any accumulated score get 0
                scores.setdefault(d, 0.0)

        # tiers read so far left too few documents: go one tier lower (the last step uses every posting)
        if tiered and restrict is not None and top_k > 0 and len(scores) < top_k:
            # same query, one more tier
            return self.rank(query, top_k, idf_threshold, min_match, tiered=True, _tier_level=_tier_level + 1)

        # champion lists left too few documents (typical for AND / phrase filters): textbook fallback to the full set
        if restrict is not None and top_k > 0 and len(scores) < top_k:
            # same query without the champion restriction
            full = self.rank(query, top_k, idf_threshold, min_match, champions=False)
            # mark the result so the caller can report it
            full.champions_fallback = True
            # exact candidates instead
            return full

        # number of documents eligible for the result list
        num_candidates = len(scores)
        # sort key: higher score first, then smaller doc_id (deterministic ties)
        key = lambda item: (item[1], -item[0])
        # bounded top-k with a heap, or a full sort
        best = heapq.nlargest(top_k, scores.items(), key=key) if top_k > 0 else sorted(scores.items(), key=key, reverse=True)
        # ranked result
        res = RankResult(best, num_candidates, {t: qtf[t] for t in kept}, dropped)
        # tiered search: report how deep it went (the bottom tier counts as one more)
        if tiered and kept:
            # tiers read, out of the stored ones plus the bottom tier
            res.tiers_used, res.tiers_total = min(_tier_level, levels) + 1, levels + 1
        # finished
        return res


# demo: python ranker.py [query ...]
if __name__ == "__main__":
    import sys
    import time

    # parser used by the demo
    from query_parser import QuerySyntaxError, parse_query

    # sample queries (command-line arguments replace them)
    queries = sys.argv[1:] or ["Hong Kong business", "Microsoft", "United States government",
                               "open source AND Linux", '"Hong Kong" business']
    # load the index once
    with IndexReader("output") as idx:
        # ranker with its own matcher
        r = Ranker(idx)
        # run every query
        for q in queries:
            # echo the input like the slide's example
            print(f"cmd> {q}")
            # malformed queries print an error instead of crashing
            try:
                # parse with the index's linguistic settings
                parsed = parse_query(q, idx.cfg)
            # syntax error
            except QuerySyntaxError as e:
                # show it
                print(f"  error: {e}")
                # next query
                continue
            # start timer
            t0 = time.time()
            # top 5 documents
            res = r.rank(parsed, top_k=5)
            # elapsed milliseconds
            ms = (time.time() - t0) * 1000
            # one "doc_id <TAB> score" line per hit, as on Slide 5
            for d, s in res.hits:
                # doc_id, score with three decimals, TREC-ID
                print(f"{d}\t{s:.3f}\t{idx.doc_label(d)}")
            # summary line
            print(f"  ({res.num_candidates:,} candidates, terms {res.terms_used}, {ms:.1f} ms)\n")
