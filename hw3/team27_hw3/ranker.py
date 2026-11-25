import heapq
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# index access layer
from index_reader import IndexReader
# Boolean / positional filter evaluation
from matcher import Matcher
# AST node types and the helper that lists ranking leaves
from query_parser import Phrase, Proximity, Query, Term, Wildcard, positive_leaves
# weighting schemes (SMART variants, BM25); the default is the Slide 4 formula
from weighting import Scheme
# postings grouped by tf, for --impact / --tiered / --tf-index
from tf_index import TfIndex, block_pairs
# cluster pruning
from clusters import Clusters


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
    # remarks for the user (fallbacks, tiers read, early termination), shown with -v
    notes: List[str] = field(default_factory=list)
    # number of postings that were scored (a machine-independent cost measure)
    postings_scored: int = 0


# TF-IDF + cosine ranking over an IndexReader, with a Matcher for the query filter
class Ranker:

    # wildcard_limit caps how many expansions of one wildcard take part in scoring
    def __init__(self, idx: IndexReader, matcher: Optional[Matcher] = None, wildcard_limit: int = 30,
                 scheme: Optional[Scheme] = None):
        # weighting scheme; None means ltc.ltc, the formula of Slide 4
        self.scheme = scheme if scheme is not None else Scheme(idx)
        # tf-ordered postings, opened on first use
        self._tfi: Optional[TfIndex] = None
        # cluster pruning tables, loaded on first use
        self._clusters: Optional[Clusters] = None
        # 1 / ||d|| by doc_id, built on first use by --impact
        self._inv_norm: Optional[List[float]] = None
        # index access
        self.idx = idx
        # Boolean / positional filter evaluation (shares the wildcard cache)
        self.matcher = matcher if matcher is not None else Matcher(idx)
        # a wildcard adds at most this many expanded terms to the score (the most frequent ones)
        self.wildcard_limit = wildcard_limit
        # term -> raw "doc doc ..." text of its champion list, loaded on first use
        self._champions: Optional[Dict[str, str]] = None
        # term -> parsed champion doc_ids, filled only for terms that were actually queried
        self._champion_sets: Dict[str, Set[int]] = {}

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

    # term -> raw id list of champions.tsv, read once per process
    def _load_champions(self) -> Dict[str, str]:
        # already loaded
        if self._champions is not None:
            # reuse
            return self._champions
        # champion lists written by prep.py: ordered by cosine contribution, or by the BM25 tf part for --scheme bm25
        path = os.path.join(self.idx.index_dir, "champions_bm25.tsv" if self.scheme.is_bm25 else "champions.tsv")
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
    # lazily opened structures of the inexact top-k methods
    # ------------------------------------------------------------------

    # tf-ordered postings (built on first use when missing)
    def tf_index(self) -> TfIndex:
        # open on first use
        if self._tfi is None:
            # builds the files first when they are missing or stale
            self._tfi = TfIndex.open(self.idx)
        # cached handle
        return self._tfi

    # cluster pruning tables
    def clusters(self) -> Clusters:
        # load on first use
        if self._clusters is None:
            # cluster files are not built automatically (they need a full pass over the collection)
            if not Clusters.available(self.idx):
                # tell the user how to build them
                raise FileNotFoundError("cluster files not found; build them with 'python prep_extra.py' first")
            # load leaders and followers
            self._clusters = Clusters(self.idx)
        # cached tables
        return self._clusters

    # ------------------------------------------------------------------
    # ranking
    # ------------------------------------------------------------------

    # rank documents for a parsed query; top_k <= 0 returns every candidate (see README for the options)
    def rank(self, query: Query, top_k: int = 10, idf_threshold: Optional[float] = None,
             min_match: int = 1, champions: bool = False, tiered: bool = False, impact: bool = False,
             cluster: Optional[Tuple[int, int]] = None, tf_index: bool = False) -> RankResult:
        # weighting scheme of this ranker
        sch = self.scheme
        # remarks collected for -v
        notes: List[str] = []
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
        # query weights and query vector length over every term present in the index (before elimination,
        # so scores stay comparable whether or not elimination is switched on)
        wq, q_norm = sch.query_weights({t: qtf[t] for t in kept})

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

        # restriction to a precomputed candidate set (None = no restriction)
        champ_docs: Optional[Set[int]] = None
        # champion lists requested
        if champions and kept:
            # union of the champion lists of the surviving terms
            champ_docs = set().union(*(self.champion_docs(t) for t in kept))
            # the BM25 lists were ordered with fixed parameters
            bm = self.idx.meta.get("champions", {}).get("schemes", {}).get("bm25", {})
            # BM25 scoring with other parameters than the lists were ordered with
            if sch.is_bm25 and (bm.get("k1"), bm.get("b")) != (sch.k1, sch.b):
                # still usable; tell the user under -v
                notes.append(f"champion lists were built with k1={bm.get('k1')}, b={bm.get('b')}, "
                             f"not k1={sch.k1}, b={sch.b}")
        # cluster pruning requested: followers of the leaders nearest to the query
        if cluster is not None and kept:
            # followers of the b2 leaders nearest to the query
            cl_docs = self.clusters().candidates({t: qtf[t] for t in kept}, cluster[0], cluster[1])
            # both restrictions together mean their intersection
            champ_docs = cl_docs if champ_docs is None else champ_docs & cl_docs

        # set version of the filter result for O(1) membership tests
        allowed: Optional[Set[int]] = set(filtered) if filtered is not None else None
        # the restriction applies on top of the filter
        if champ_docs is not None:
            # intersect with the filter, or use the restriction alone
            allowed = champ_docs if allowed is None else allowed & champ_docs

        # tiered index: read tier 1 of every term, go one tier down only while fewer than top_k documents were found
        tier_limited = False
        # tiered search needs query terms and a finite k
        if tiered and kept and top_k > 0:
            # tf-ordered postings with the tier thresholds
            tfi = self.tf_index()
            # documents collected from the tiers read so far
            found: Set[int] = set()
            # every term's blocks, read once
            blocks = {t: tfi.blocks(t) for t in kept}
            # tiers from best to worst
            for tier in range(tfi.num_tiers):
                # every surviving query term
                for t in kept:
                    # doc arrays of this term in this tier
                    for docs in tfi.tier_docs(t, tier, blocks[t]):
                        # add them, restricted to the filter when there is one
                        found.update(docs if allowed is None else allowed.intersection(docs))
                # enough documents: stop going down
                if len(found) >= top_k:
                    # leave the tier loop
                    break
            # report how deep the search went
            notes.append(f"tiered index: read {tier + 1} of {tfi.num_tiers} tiers, {len(found):,} documents")
            # all tiers read means every posting was seen: nothing is pruned
            if tier + 1 < tfi.num_tiers:
                # score only the documents found in the tiers that were read
                allowed, tier_limited = found, True

        # min_match only applies to free text (Boolean results are already exact)
        use_min_match = query.mode == "free" and min_match > 1 and bool(kept)
        # number of postings scored
        scored = 0
        # doc_id -> similarity
        scores: Dict[int, float] = {}
        # impact ordering: tf-ordered postings, max-score bounds, early termination
        if impact and kept:
            # final scores and postings scored
            scores, scored = self._accumulate_impact(kept, wq, q_norm, allowed, top_k, use_min_match, min_match, notes)
        # exact term-at-a-time scoring
        else:
            # doc_id -> accumulated dot product
            acc: Dict[int, float] = {}
            # doc_id -> number of query terms it contains (for min_match)
            hits_per_doc: Counter = Counter()
            # postings come from the tf index when one of its users is switched on, else from postings.txt
            use_tfi = tf_index or tiered or cluster is not None
            # term-at-a-time accumulation
            for t in kept:
                # query-side weight of the term
                wq_t = wq[t]
                # document-side weight function of the term
                wd = sch.doc_weight_fn(t)
                # postings of the term, already restricted when they come from the tf index
                if use_tfi:
                    # (doc_id, tf) from tf_index.bin, restricted to `allowed`
                    pairs = self.tf_index().doc_tfs(t, allowed)
                # default source
                else:
                    # (doc_id, tf) parsed from postings.txt
                    pairs = self.idx.get_doc_tfs(t)
                # walk the term's postings
                for d, tf in pairs:
                    # skip documents outside the filter / candidate set
                    if allowed is not None and d not in allowed:
                        # next posting
                        continue
                    # add w(t, q) * w(t, d)
                    acc[d] = acc.get(d, 0.0) + wq_t * wd(d, tf)
                    # one more query term found in this document
                    hits_per_doc[d] += 1
                    # count the posting
                    scored += 1

            # keep documents containing enough query terms
            if use_min_match:
                # cannot demand more terms than the query has
                need = min(min_match, len(kept))
                # drop documents with fewer matching terms
                acc = {d: s for d, s in acc.items() if hits_per_doc[d] >= need}

            # document vector lengths under this scheme (all 1 without cosine normalization)
            d_norms = sch.doc_norms()
            # normalize every accumulator
            for d, dot in acc.items():
                # document vector length
                d_norm = d_norms[d]
                # zero-length vectors cannot be normalized
                if d_norm > 0.0 and q_norm > 0.0:
                    # cosine similarity
                    scores[d] = dot / (q_norm * d_norm)

        # filtered queries keep matching documents that scored 0 (e.g. "NOT x", or docs matched only via OR NOT)
        if filtered is not None:
            # every filtered document that survived the restriction
            keep = allowed if (champ_docs is not None or tier_limited) else None
            # filtered documents, restricted when a candidate restriction is active
            for d in (filtered if keep is None else (x for x in filtered if x in keep)):
                # documents without any accumulated score get 0
                scores.setdefault(d, 0.0)

        # the restriction left too few documents (typical for AND / phrase filters): textbook fallback
        if (champ_docs is not None or tier_limited) and top_k > 0 and len(scores) < top_k:
            # cluster pruning first widens the search: twice as many leaders, until all of them are used
            if cluster is not None and cluster[1] < len(self.clusters().leaders):
                # same query with b2 doubled
                full = self.rank(query, top_k, idf_threshold, min_match, champions, tiered, impact,
                                 (cluster[0], cluster[1] * 2), tf_index)
            # otherwise the same query without any restriction
            else:
                # exact search (impact ordering stays on: it is exact)
                full = self.rank(query, top_k, idf_threshold, min_match, False, False, impact, None, tf_index)
            # mark the result so the caller can report it
            full.champions_fallback = True
            # postings scored by the failed attempt were work too
            full.postings_scored += scored
            # exact candidates instead
            return full

        # number of documents eligible for the result list
        num_candidates = len(scores)
        # sort key: higher score first, then smaller doc_id (deterministic ties)
        key = lambda item: (item[1], -item[0])
        # bounded top-k with a heap, or a full sort
        best = heapq.nlargest(top_k, scores.items(), key=key) if top_k > 0 else sorted(scores.items(), key=key, reverse=True)
        # ranked result
        return RankResult(best, num_candidates, {t: qtf[t] for t in kept}, dropped, notes=notes, postings_scored=scored)

    # Impact-ordered scoring (IIR 7.1.5 with a safe stopping rule). Terms are processed in decreasing order of the
    # largest contribution they can make, their postings in decreasing tf. Once the k-th best score so far exceeds
    # everything a not yet seen document could still collect, no new accumulators are created: the remaining
    # postings are only looked up for the surviving documents. Returns (doc_id -> final score, postings scored).
    def _accumulate_impact(self, kept: List[str], wq: Dict[str, float], q_norm: float, allowed: Optional[Set[int]],
                           top_k: int, use_min_match: bool, min_match: int, notes: List[str]):
        # weighting scheme of this ranker
        sch = self.scheme
        # tf-ordered postings with the stored bounds
        tfi = self.tf_index()
        # which stored bound fits the scheme ("ltc", "bm25" or None)
        kind = sch.bound_kind(tfi.meta)
        # vector lengths of the scheme
        d_norms = sch.doc_norms()
        # 1 / ||d||, so a posting costs one multiplication
        if self._inv_norm is None or len(self._inv_norm) != len(d_norms):
            # reciprocal of every length (0 for empty documents), cached on the ranker
            self._inv_norm = [1.0 / v if v > 0.0 else 0.0 for v in d_norms]
        # local name for speed
        inv_norm = self._inv_norm
        # query normalization folded into every contribution
        inv_q = 1.0 / q_norm if q_norm > 0.0 else 0.0
        # early termination needs a bound, a finite k and no min_match (which changes who may enter the top k)
        safe = kind is not None and top_k > 0 and not use_min_match
        # early termination not possible
        if not safe:
            # explain under -v why no early termination happens
            notes.append("impact ordering: no early termination ("
                         + ("no stored bound for this scheme" if kind is None else "top-k = all or min-match") + ")")
        # per term: largest possible contribution, weight function, blocks, per-block bound factor
        recs = []
        # every surviving query term
        for t in kept:
            # its tf blocks, highest tf first
            blocks = tfi.blocks(t)
            # term without postings
            if not blocks:
                # nothing to score
                continue
            # cosine bound
            if kind == "ltc":
                # contribution = wq * idf * (1 + log tf) / (|q| |d|); the stored bound covers (1 + log tf) / |d|
                factor = wq[t] * sch.idf(t, "t") * inv_q
            # BM25 bound
            elif kind == "bm25":
                # contribution = qtf * idf * tf part; the stored bound covers the tf part
                factor = wq[t] * sch.bm25_idf(t)
            # no usable bound: early termination is off anyway
            else:
                # bounds are not used
                factor = 0.0
            # bound of every block (suffix maxima, so bounds[0] is the term's largest possible contribution)
            bounds = [factor * (b.bound_ltc if kind == "ltc" else b.bound_bm25) for b in blocks]
            # one record per term
            recs.append((bounds[0], t, blocks, bounds))
        # the term that can contribute most goes first (highest idf first, for equal query tf)
        recs.sort(key=lambda r: -r[0])
        # rest[i] = most a document can collect from terms i, i+1, ...
        rest = [0.0] * (len(recs) + 1)
        # from the last term backwards
        for i in range(len(recs) - 1, -1, -1):
            # suffix sum of the per-term maxima
            rest[i] = rest[i + 1] + recs[i][0]

        # doc_id -> score so far (already normalized)
        acc: Dict[int, float] = {}
        # doc_id -> number of query terms seen (for min_match)
        hits: Counter = Counter()
        # postings scored
        scored = 0
        # False while new documents may still enter the top k
        closed = False
        # (term position, tf) where the search closed, for the -v note
        closed_at = None
        # terms in decreasing order of their largest contribution
        for i, (_, t, blocks, bounds) in enumerate(recs):
            # document-side weight function of the term
            wd = sch.doc_weight_fn(t)
            # query weight divided by the query length
            c = wq[t] * inv_q
            # after closing: drop accumulators that cannot reach the current k-th score any more
            if closed and len(acc) > top_k:
                # current k-th best score
                theta = heapq.nlargest(top_k, acc.values())[-1]
                # keep documents that could still reach it with the remaining terms
                acc = {d: s for d, s in acc.items() if s + rest[i] >= theta}
            # blocks of the term, highest tf first
            for blk, bound in zip(blocks, bounds):
                # test the stopping rule before every large block (small blocks are cheaper to score than to test)
                if safe and not closed and len(acc) >= top_k and len(blk.docs) >= 256:
                    # current k-th best score
                    theta = heapq.nlargest(top_k, acc.values())[-1]
                    # a document not seen so far can collect at most bound + rest[i + 1]
                    if theta > bound + rest[i + 1]:
                        # no new document can enter the top k from here on
                        closed, closed_at = True, (i, blk.tf)
                # closed: update existing accumulators only
                if closed:
                    # only documents that already have an accumulator
                    for d, tf in block_pairs(blk, acc):
                        # add the normalized contribution
                        acc[d] += c * wd(d, tf) * inv_norm[d]
                        # count the posting
                        scored += 1
                # still open: every allowed document of the block
                else:
                    # documents of the block, restricted to the filter / candidate set
                    for d, tf in block_pairs(blk, allowed):
                        # add the normalized contribution, creating the accumulator if needed
                        acc[d] = acc.get(d, 0.0) + c * wd(d, tf) * inv_norm[d]
                        # count the posting
                        scored += 1
                        # min_match needs the number of matching terms
                        if use_min_match:
                            # one more query term in this document
                            hits[d] += 1
        # report where the search closed
        if closed_at is not None:
            # term number, term, tf limit and accumulators left
            notes.append(f"impact ordering: no new documents after term {closed_at[0] + 1} of {len(recs)} "
                         f"({recs[closed_at[0]][1]!r}, tf < {closed_at[1] + 1}); {len(acc):,} accumulators left")
        # keep documents containing enough query terms
        if use_min_match:
            # cannot demand more terms than the query has
            need = min(min_match, len(kept))
            # drop documents with fewer matching terms
            acc = {d: s for d, s in acc.items() if hits[d] >= need}
        # documents with a zero-length vector cannot be ranked (same rule as the exact path)
        return {d: s for d, s in acc.items() if inv_norm[d] > 0.0 and inv_q > 0.0}, scored


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
