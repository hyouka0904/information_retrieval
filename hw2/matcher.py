import heapq
import math
import re
from typing import Dict, Iterable, List, Optional

# index access layer
from index_reader import IndexReader
# AST node types produced by query_parser
from query_parser import And, Node, Not, Or, Phrase, Proximity, Query, Term, Wildcard
# surface forms for wildcards on the original words
from spell import SurfaceVocab, wildcard_regex
# k-gram index for general wildcard patterns
from kgram import KGramIndex


# Boolean / positional evaluation of a query filter on top of an IndexReader
class Matcher:

    # per-query and cross-query caches, counters and the lazily loaded surface vocabulary
    def __init__(self, idx: IndexReader):
        # index access (postings, vocabulary, N)
        self.idx = idx
        # pattern -> expanded index terms; kept across queries because the vocabulary never changes
        self._wildcard_cache: Dict[str, List[str]] = {}
        # term -> sorted doc_ids, cleared at the start of every query
        self._ids_cache: Dict[str, List[int]] = {}
        # term -> PostingEntry list with positions, cleared at the start of every query
        self._pos_cache: Dict[str, list] = {}
        # counters for the last query (reported by the CLI / documentation)
        self.stats: Dict[str, int] = {}
        # surface forms of the collection (surface_vocab.tsv), loaded on first use
        self._surface: Optional[SurfaceVocab] = None
        # True once surface_vocab.tsv was looked for
        self._surface_loaded = False
        # k-gram index over index terms and surface forms (kgram.bin), opened on first use
        self._kgram: Optional[KGramIndex] = None
        # True once kgram.bin was looked for
        self._kgram_loaded = False
        # set to False to force the original dictionary scan (used to cross-check the k-gram path)
        self.use_kgram = True

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    # sorted doc_ids satisfying query.filter, or None when the query has no filter (free text)
    def match(self, query: Query) -> Optional[List[int]]:
        # postings fetched for the previous query are not reused (bounded memory)
        self._ids_cache.clear()
        # same for positional postings
        self._pos_cache.clear()
        # fresh counters for this query
        self.stats = {"comparisons": 0, "skips": 0, "postings_read": 0}
        # free text without phrase / proximity: every document containing a query term is a candidate
        if query.filter is None:
            # the ranker builds candidates from the scoring terms itself
            return None
        # evaluate the filter tree
        return self.evaluate(query.filter)

    # every index term matching pattern (* = any run, ? = one character), plus the terms of matching surface forms, sorted
    def expand_wildcard(self, pattern: str) -> List[str]:
        # previously expanded pattern
        if pattern in self._wildcard_cache:
            # reuse the cached expansion
            return self._wildcard_cache[pattern]
        # position of the first wildcard character
        first_wild = min(i for i, c in enumerate(pattern) if c in "*?")
        # literal text before it; every match must start with this prefix
        prefix = pattern[:first_wild]
        # candidates from the k-gram index for everything but a pure prefix pattern (None = not usable)
        grams = self.kgram().candidates(pattern) if (pattern != prefix + "*" and self.kgram() is not None) else None
        # pure prefix pattern ("comput*"): the sorted vocabulary answers it directly
        if pattern == prefix + "*":
            # contiguous run of terms with this prefix
            terms = self.idx.terms_with_prefix(prefix)
        # general pattern with the k-gram index (IIR 3.2.2): intersect the gram lists, then post-filter,
        # because sharing every k-gram is necessary but not sufficient ("red*" also proposes "retired")
        elif grams is not None:
            # same regexes as the scan below, applied to the few candidates only
            regex = re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern))
            # dictionary in sorted order: a term id is its position
            all_terms = self.idx.all_terms()
            # index terms that really match
            matched = {all_terms[i] for i in grams[0] if regex.fullmatch(all_terms[i])}
            # surface forms that really match contribute the term they were stemmed to
            vocab = self.surface_vocab()
            # surface-form regex (same rules as SurfaceVocab's wildcard matching)
            s_regex = wildcard_regex(pattern)
            # add the stemmed term of every surface candidate that really matches
            matched.update(vocab.terms[k] for k in grams[1] if s_regex.fullmatch(vocab.surfaces[k]))
            # remember and return; the surface forms are already included
            self._wildcard_cache[pattern] = sorted(matched)
            # sorted expansion
            return self._wildcard_cache[pattern]
        # general pattern ("wom?n", "*ology", "c*t*r")
        else:
            # translate to a regex: * -> .*, ? -> ., everything else literal
            regex = re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern))
            # only terms with the literal prefix can match (the whole vocabulary if the prefix is empty)
            pool = self.idx.terms_with_prefix(prefix) if prefix else self.idx.all_terms()
            # keep terms the regex matches completely
            terms = [t for t in pool if regex.fullmatch(t)]
        # the same pattern on the original words ("*ing" finds "running" -> "run"), when the vocabulary exists
        vocab = self.surface_vocab()
        # surface forms are available
        if vocab is not None:
            # union of both matches, sorted
            terms = sorted(set(terms).union(vocab.terms_for_pattern(pattern)))
        # remember for later queries
        self._wildcard_cache[pattern] = terms
        # sorted list of matching terms
        return terms

    # surface_vocab.tsv, loaded on first use; None when it is missing or not described in meta.json
    def surface_vocab(self) -> Optional[SurfaceVocab]:
        # first call: look for surface_vocab.tsv once
        if not self._surface_loaded:
            # None when the file is missing or meta.json does not describe it (wildcards then match index terms only)
            self._surface = SurfaceVocab.load(self.idx.index_dir) if self.idx.meta.get("surface_vocab") else None
            # do not look again
            self._surface_loaded = True
        # loaded vocabulary or None
        return self._surface

    # kgram.bin, opened on first use (built then, about 3 seconds, when it is missing); None when switched off
    def kgram(self) -> Optional[KGramIndex]:
        # switched off (cross-checking the original scan)
        if not self.use_kgram:
            # callers fall back to the scan
            return None
        # first call
        if not self._kgram_loaded:
            # look only once
            self._kgram_loaded = True
            # surface vocabulary, part of the k-gram index's strings
            vocab = self.surface_vocab()
            # no surface vocabulary: index terms only
            surfaces = vocab.surfaces if vocab is not None else []
            # missing, or built for another index / surface vocabulary
            if not KGramIndex.available(self.idx, len(surfaces)):
                import sys
                # k-gram builder
                import kgram
                # tell the user why this query is slower
                print("# building the k-gram index (one time, a few seconds) ...", file=sys.stderr, flush=True)
                # build kgram.bin / kgram.json
                kgram.build(self.idx, surfaces)
            # open it
            self._kgram = KGramIndex(self.idx.index_dir)
        # cached index
        return self._kgram

    # ------------------------------------------------------------------
    # postings with per-query caching
    # ------------------------------------------------------------------

    # sorted doc_ids of a term, read once per query
    def _doc_ids(self, term: str) -> List[int]:
        # cached result for this query
        ids = self._ids_cache.get(term)
        # first time this term is needed in this query
        if ids is None:
            # doc_ids only, no positions parsed
            ids = [d for d, _ in self.idx.get_doc_tfs(term)]
            # cache it
            self._ids_cache[term] = ids
            # count postings read from disk
            self.stats["postings_read"] += len(ids)
        # sorted doc_ids
        return ids

    # PostingEntry list with positions of a term, read once per query
    def _positions(self, term: str) -> list:
        # cached result for this query
        plist = self._pos_cache.get(term)
        # first time positions of this term are needed
        if plist is None:
            # full PostingEntry list with positions
            plist = self.idx.get_postings(term)
            # cache it
            self._pos_cache[term] = plist
            # count postings read from disk
            self.stats["postings_read"] += len(plist)
        # PostingEntry list sorted by doc_id
        return plist

    # ------------------------------------------------------------------
    # list algebra on sorted doc_id lists
    # ------------------------------------------------------------------

    # linear merge with skip pointers every floor(sqrt(len)) entries on both lists (textbook Fig. 2.10)
    def _intersect(self, a: List[int], b: List[int]) -> List[int]:
        # skip distance on list a (at least 1)
        skip_a = max(1, int(math.isqrt(len(a))))
        # skip distance on list b (at least 1)
        skip_b = max(1, int(math.isqrt(len(b))))
        # matching doc_ids
        out: List[int] = []
        # cursors into a and b
        i = j = 0
        # local copies of the counters (faster than dict updates in the loop)
        comparisons = skips = 0
        # stop when either list is exhausted
        while i < len(a) and j < len(b):
            # one doc_id comparison
            comparisons += 1
            # same document in both lists
            if a[i] == b[j]:
                # keep it
                out.append(a[i])
                # advance both cursors
                i += 1
                # advance both cursors
                j += 1
            # a is behind
            elif a[i] < b[j]:
                # a skip pointer exists at i (every skip_a entries) and its target does not overshoot b[j]
                if i % skip_a == 0 and i + skip_a < len(a) and a[i + skip_a] <= b[j]:
                    # follow skips as long as they stay <= b[j]
                    while i % skip_a == 0 and i + skip_a < len(a) and a[i + skip_a] <= b[j]:
                        # jump ahead by one skip
                        i += skip_a
                        # count the skip
                        skips += 1
                # no useful skip
                else:
                    # step by one
                    i += 1
            # b is behind
            else:
                # a skip pointer exists at j and its target does not overshoot a[i]
                if j % skip_b == 0 and j + skip_b < len(b) and b[j + skip_b] <= a[i]:
                    # follow skips as long as they stay <= a[i]
                    while j % skip_b == 0 and j + skip_b < len(b) and b[j + skip_b] <= a[i]:
                        # jump ahead by one skip
                        j += skip_b
                        # count the skip
                        skips += 1
                # no useful skip
                else:
                    # step by one
                    j += 1
        # add this merge's comparisons to the query total
        self.stats["comparisons"] += comparisons
        # add this merge's skips to the query total
        self.stats["skips"] += skips
        # sorted intersection
        return out

    # k-way merge of sorted lists without duplicates
    @staticmethod
    def _union(lists: Iterable[List[int]]) -> List[int]:
        # merged, deduplicated doc_ids
        out: List[int] = []
        # k-way merge of sorted lists keeps the output sorted
        for d in heapq.merge(*lists):
            # skip duplicates (equal ids arrive next to each other)
            if not out or out[-1] != d:
                # new doc_id
                out.append(d)
        # sorted union
        return out

    # doc_ids of a that are not in b (both sorted)
    @staticmethod
    def _difference(a: List[int], b: List[int]) -> List[int]:
        # set of documents to remove
        drop = set(b)
        # keep order of a
        return [d for d in a if d not in drop]

    # every doc_id 1..N, built once
    def _universe(self) -> List[int]:
        # every doc_id of the collection
        return list(range(1, self.idx.N + 1))

    # ------------------------------------------------------------------
    # node evaluation
    # ------------------------------------------------------------------

    # sorted doc_ids matching one AST node
    def evaluate(self, node: Node) -> List[int]:
        # plain term
        if isinstance(node, Term):
            # its postings
            return self._doc_ids(node.term)
        # exact phrase
        if isinstance(node, Phrase):
            # positional match
            return self._phrase(node)
        # proximity pair
        if isinstance(node, Proximity):
            # positional window match
            return self._proximity(node)
        # wildcard pattern
        if isinstance(node, Wildcard):
            # documents containing any expanded term
            return self._union(self._doc_ids(t) for t in self.expand_wildcard(node.pattern))
        # conjunction
        if isinstance(node, And):
            # intersect / subtract
            return self._and(node)
        # disjunction
        if isinstance(node, Or):
            # union of every child
            return self._union(self.evaluate(c) for c in node.children)
        # negation outside an AND
        if isinstance(node, Not):
            # complement against the whole collection
            return self._difference(self._universe(), self.evaluate(node.child))
        # parser produced something unexpected
        raise TypeError(f"unknown node {node!r}")

    # AND: positive children intersected shortest first, NOT children subtracted
    def _and(self, node: And) -> List[int]:
        # results of the non-negated children
        positives = [self.evaluate(c) for c in node.children if not isinstance(c, Not)]
        # results of the negated children (their inner expression)
        negatives = [self.evaluate(c.child) for c in node.children if isinstance(c, Not)]
        # only NOTs, e.g. "NOT a AND NOT b": start from every document
        if not positives:
            # the whole collection
            positives = [self._universe()]
        # shortest list first keeps intermediate results small
        positives.sort(key=len)
        # running intersection
        result = positives[0]
        # intersect with the remaining lists
        for other in positives[1:]:
            # nothing can match any more
            if not result:
                # stop early
                break
            # pairwise merge with skip pointers
            result = self._intersect(result, other)
        # remove documents matching any negated child
        if negatives and result:
            # subtract their union
            result = self._difference(result, self._union(negatives))
        # sorted doc_ids
        return result

    # phrase: rarest term first, then every other term must sit at its offset
    def _phrase(self, node: Phrase) -> List[int]:
        # positional postings of every distinct phrase term
        plists = {t: self._positions(t) for t, _ in node.terms}
        # a missing term means no document can contain the phrase
        if any(not plists[t] for t, _ in node.terms):
            # empty result
            return []
        # rarest term first: fewest candidate documents to carry along
        order = sorted(node.terms, key=lambda x: len(plists[x[0]]))
        # rarest term and its offset inside the phrase
        first_term, first_off = order[0]
        # doc_id -> possible phrase start positions (position of the term minus its offset)
        cand = {p.doc_id: {pos - first_off for pos in p.positions} for p in plists[first_term]}
        # narrow down with every other term
        for term, off in order[1:]:
            # candidates that survive this term
            nxt: Dict[int, set] = {}
            # documents containing this term
            for p in plists[term]:
                # start positions still possible in this document
                starts = cand.get(p.doc_id)
                # document already ruled out
                if starts is None:
                    # next posting
                    continue
                # starts where this term sits at its expected offset
                ok = starts.intersection(pos - off for pos in p.positions)
                # at least one start survives
                if ok:
                    # keep the document with the surviving starts
                    nxt[p.doc_id] = ok
            # survivors become the new candidates
            cand = nxt
            # nothing left
            if not cand:
                # stop early
                break
        # sorted doc_ids that contain the full phrase
        return sorted(cand)

    # proximity: documents where the two terms occur within k positions, either order
    def _proximity(self, node: Proximity) -> List[int]:
        # positional postings of the left term
        left = self._positions(node.left)
        # positional postings of the right term
        right = self._positions(node.right)
        # doc_id -> positions of the right term
        right_pos = {p.doc_id: p.positions for p in right}
        # documents containing both terms (merged with skip pointers)
        both = self._intersect([p.doc_id for p in left], [p.doc_id for p in right])
        # doc_id -> positions of the left term
        left_pos = {p.doc_id: p.positions for p in left}
        # documents with at least one close pair
        out: List[int] = []
        # check each shared document
        for d in both:
            # same term on both sides ("a /3 a"): needs two different occurrences close together
            if node.left == node.right:
                # sorted positions of the term
                ps = left_pos[d]
                # some consecutive pair within k
                if any(b - a <= node.k for a, b in zip(ps, ps[1:])):
                    # keep the document
                    out.append(d)
            # different terms: two-pointer walk over both sorted position lists
            elif self._close_pair(left_pos[d], right_pos[d], node.k):
                # keep the document
                out.append(d)
        # sorted doc_ids
        return out

    # True if some position of a and some position of b are at most k apart (two pointers)
    @staticmethod
    def _close_pair(a: List[int], b: List[int], k: int) -> bool:
        # cursors into the two sorted position lists
        i = j = 0
        # walk until one list is exhausted
        while i < len(a) and j < len(b):
            # found a pair within the window
            if abs(a[i] - b[j]) <= k:
                # one pair is enough
                return True
            # advance the smaller position; the other side cannot get closer by moving forward
            if a[i] < b[j]:
                # next position of a
                i += 1
            # b is smaller
            else:
                # next position of b
                j += 1
        # no pair within k
        return False


# demo: python matcher.py [query ...]
if __name__ == "__main__":
    import sys
    import time

    # parser used by the demo
    from query_parser import QuerySyntaxError, parse_query

    # sample queries (command-line arguments replace them)
    queries = sys.argv[1:] or [
        "open source AND Linux",
        "software engineer OR project manager",
        '"Hong Kong" business',
        '"United States of America"',
        "hong /3 kong",
        "comput* AND NOT (windows OR mac)",
        "NOT microsoft",
    ]
    # load the index once
    with IndexReader("output") as idx:
        # matcher on top of it
        m = Matcher(idx)
        # run every query
        for q in queries:
            # echo the input
            print(f"> {q}")
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
            # evaluate the filter
            docs = m.match(parsed)
            # elapsed milliseconds
            ms = (time.time() - t0) * 1000
            # free text without a filter
            if docs is None:
                # nothing to report here; the ranker handles it
                print(f"  filter: (none)  scoring: {', '.join(map(str, parsed.scoring))}")
            # filtered result
            else:
                # parsed filter, result count, first ids, counters and time
                print(f"  filter: {parsed.filter}\n  {len(docs):,} docs, first {docs[:8]}  {m.stats}  {ms:.1f} ms")
