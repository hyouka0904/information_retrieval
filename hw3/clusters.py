"""Cluster pruning index construction and query-time candidate selection.

Selects k = round(sqrt(N)) = 177 random leader docs (fixed seed) and assigns every 
doc in the collection to its MAX_B1 = 3 nearest leaders under ltc cosine similarity. 
At query, ranks the 177 leaders against the query using a compact in-memory inverted 
index over leader vectors and returns the union of followers attached to the closest b2 leaders.
"""

import array
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# file names inside the index folder: leader vectors
LEADERS_FILE = "cluster_leaders.tsv"
# nearest leaders of every document
FOLLOWERS_FILE = "cluster_followers.tsv"
# description of the build
META_FILE = "clusters.json"
# every document is attached to this many nearest leaders at build time (b1 of IIR 7.1.6 can be 1..this)
MAX_B1 = 3
# seed of the random leader choice, so a rebuild gives the same files
DEFAULT_SEED = 27


# Cluster pruning (IIR 7.1.6): sqrt(N) random leaders, every document follows its nearest leaders.
# At query time only the followers of the leaders closest to the query are scored.
#   cluster_leaders.tsv    leader doc_id <TAB> term:weight ...   (unit-length ltc vector of the leader)
#   cluster_followers.tsv  doc_id <TAB> leader leader leader      (nearest first, at most MAX_B1)
class Clusters:

    # load leader vectors and follower lists of an existing build
    def __init__(self, idx):
        # index access (tf weight, idf)
        self.idx = idx
        # description written by build()
        with open(os.path.join(idx.index_dir, META_FILE), encoding="utf-8") as f:
            # parsed clusters.json
            self.meta: dict = json.load(f)
        # term -> [(leader position, weight)], an inverted index over the leader vectors
        self._by_term: Dict[str, List[Tuple[int, float]]] = {}
        # leader doc_ids in file order
        self.leaders: List[int] = []
        # LF-only line splitting, matching how the file was written
        with open(os.path.join(idx.index_dir, LEADERS_FILE), encoding="utf-8", newline="\n") as f:
            # one leader per line
            for line in f:
                # doc_id followed by term:weight cells
                cells = line.rstrip("\n").split("\t")
                # position of this leader
                pos = len(self.leaders)
                # remember its doc_id
                self.leaders.append(int(cells[0]))
                # every component of its vector
                for cell in cells[1:]:
                    # a term may itself contain ':', so split at the last one
                    term, w = cell.rsplit(":", 1)
                    # add to the term's list
                    self._by_term.setdefault(term, []).append((pos, float(w)))
        # rank r (0 = nearest) -> leader doc_id -> follower doc_ids
        self._followers: List[Dict[int, List[int]]] = [{} for _ in range(MAX_B1)]
        # LF-only line splitting, matching how the file was written
        with open(os.path.join(idx.index_dir, FOLLOWERS_FILE), encoding="utf-8", newline="\n") as f:
            # one document per line
            for line in f:
                # doc_id and its leaders, nearest first
                d_s, ids = line.rstrip("\n").split("\t")
                # rank of each leader for this document
                for r, leader in enumerate(ids.split()):
                    # the document follows this leader at rank r
                    self._followers[r].setdefault(int(leader), []).append(int(d_s))

    # True when the cluster files exist and describe the postings file on disk
    @staticmethod
    def available(idx) -> bool:
        # the three file paths
        paths = [os.path.join(idx.index_dir, n) for n in (LEADERS_FILE, FOLLOWERS_FILE, META_FILE)]
        # any of them missing
        if not all(os.path.exists(p) for p in paths):
            # needs a build
            return False
        # read clusters.json
        with open(paths[2], encoding="utf-8") as f:
            # built from a postings file of the same size as the current one
            return json.load(f).get("postings_bytes") == idx.meta["postings_bytes"]

    # leaders ordered by cosine similarity to the query (ltc query weights), nearest first; ties by doc_id
    def nearest_leaders(self, qtf: Dict[str, int]) -> List[int]:
        # dot product of the query with every leader
        sims = [0.0] * len(self.leaders)
        # every query term
        for t, c in qtf.items():
            # ltc query weight (the query length is the same for every leader, so it is left out)
            wq = self.idx.tf_weight(c) * self.idx.idf(t)
            # leaders whose vector contains the term
            for pos, wl in self._by_term.get(t, ()):
                # add w(t, q) * w(t, leader)
                sims[pos] += wq * wl
        # leader positions by decreasing similarity, smaller doc_id on ties
        order = sorted(range(len(self.leaders)), key=lambda p: (-sims[p], self.leaders[p]))
        # as leader doc_ids
        return [self.leaders[p] for p in order]

    # documents to score: followers of the b2 leaders nearest to the query, using each document's b1 nearest leaders
    def candidates(self, qtf: Dict[str, int], b1: int, b2: int) -> Set[int]:
        # candidate documents
        out: Set[int] = set()
        # the b2 leaders nearest to the query
        for leader in self.nearest_leaders(qtf)[:b2]:
            # documents for which this leader is among their b1 nearest
            for r in range(min(b1, MAX_B1)):
                # add the followers at rank r
                out.update(self._followers[r].get(leader, ()))
        # union of the followers
        return out


# build the cluster files from the tf index and doc_norms.tsv; returns clusters.json's content
def build(idx, tfi, seed: int = DEFAULT_SEED, num_leaders: Optional[int] = None) -> dict:
    # start timer
    t0 = time.time()
    # number of documents
    n = idx.N
    # documents with at least one term can be leaders or followers
    docs = [d for d in range(1, n + 1) if idx.doc_norms[d] > 0.0]
    # sqrt(N) leaders unless told otherwise
    k = num_leaders or int(round(math.sqrt(n)))
    # random leaders with a fixed seed, sorted by doc_id
    leaders = sorted(random.Random(seed).sample(docs, k))
    # leader doc_id -> position
    leader_pos = {d: p for p, d in enumerate(leaders)}
    # fast membership test
    leader_set = set(leaders)
    # 1 / ||d|| by doc_id
    inv_norm = [1.0 / v if v > 0.0 else 0.0 for v in idx.doc_norms]

    # pass 1: unit-length ltc vector of every leader, as term -> [(leader position, weight)]
    by_term: Dict[str, List[Tuple[int, float]]] = {}
    # leader position -> [(term, weight)], written to cluster_leaders.tsv
    vectors: List[List[Tuple[str, float]]] = [[] for _ in leaders]
    # every index term, in dictionary order
    terms = idx.all_terms()
    # every term
    for term in terms:
        # idf of the term
        idf = idx.idf(term)
        # terms in every document carry no weight
        if idf == 0.0:
            # skip
            continue
        # every tf block of the term
        for blk in tfi.blocks(term):
            # ltc weight before normalization
            w = idx.tf_weight(blk.tf) * idf
            # leaders in this block
            for d in leader_set.intersection(blk.docs):
                # position of the leader
                p = leader_pos[d]
                # unit-length weight, indexed by term
                by_term.setdefault(term, []).append((p, w * inv_norm[d]))
                # same weight, indexed by leader
                vectors[p].append((term, w * inv_norm[d]))

    # pass 2: cosine similarity of every document with every leader, one dense column per leader
    sims = [array.array("d", bytes(8 * (n + 1))) for _ in leaders]
    # only terms that occur in some leader can contribute
    for term, entries in by_term.items():
        # idf of the term
        idf = idx.idf(term)
        # every tf block of the term
        for blk in tfi.blocks(term):
            # ltc weight before normalization
            w = idx.tf_weight(blk.tf) * idf
            # documents of the block
            docs_blk = blk.docs
            # every leader containing the term
            for p, wl in entries:
                # similarity column of this leader
                col = sims[p]
                # w(t, d) * w(t, leader) without the document's norm
                c = w * wl
                # every document of the block
                for d in docs_blk:
                    # add the normalized product
                    col[d] += c * inv_norm[d]

    # nearest MAX_B1 leaders of every document (higher similarity first, smaller leader doc_id on ties)
    with open(os.path.join(idx.index_dir, FOLLOWERS_FILE), "w", encoding="utf-8", newline="\n") as f:
        # every document with terms
        for d in docs:
            # (negative similarity, leader) for leaders sharing a term with the document
            scored = [(-sims[p][d], leaders[p]) for p in range(k) if sims[p][d] > 0.0]
            # nearest first, smaller leader doc_id on ties
            scored.sort()
            # a leader always follows itself first (its similarity with itself is 1)
            if d in leader_set:
                # put itself in front and drop the duplicate
                scored = [(-2.0, d)] + [s for s in scored if s[1] != d]
            # documents sharing no term with any leader get no line
            if scored:
                # doc_id and its nearest MAX_B1 leaders
                f.write(f"{d}\t{' '.join(str(l) for _, l in scored[:MAX_B1])}\n")
    # leader vectors, LF-only
    with open(os.path.join(idx.index_dir, LEADERS_FILE), "w", encoding="utf-8", newline="\n") as f:
        # every leader in position order
        for p, d in enumerate(leaders):
            # doc_id and term:weight cells (repr keeps every float digit)
            f.write(f"{d}\t" + "\t".join(f"{t}:{w!r}" for t, w in vectors[p]) + "\n")
    # content of clusters.json
    info = {"postings_bytes": idx.meta["postings_bytes"], "num_leaders": k, "max_b1": MAX_B1, "seed": seed,
            "similarity": "ltc cosine", "build_seconds": round(time.time() - t0, 1)}
    # written last, so it always describes the files next to it
    with open(os.path.join(idx.index_dir, META_FILE), "w", encoding="utf-8", newline="\n") as f:
        # readable JSON
        json.dump(info, f, indent=2)
        # trailing newline
        f.write("\n")
    # description of the build
    return info
