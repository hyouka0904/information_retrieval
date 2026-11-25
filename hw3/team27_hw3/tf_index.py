"""Impact-ordered binary posting index without positions (tf_index.bin/.dir/.json).

Groups each term's postings by tf in decreasing order, storing sorted doc_id arrays per block, along precomputed suffix-maximum score upper bounds for ltc cosine and BM25.
Serves as the storage engine for --tf-index (fast exact scoring), --impact (lossless early termination), and --tiered.
"""

import array
import bisect
import json
import os
import struct
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

# file names inside the index folder: records of every term
BIN_FILE = "tf_index.bin"
# byte offset of every term's record
DIR_FILE = "tf_index.dir"
# description of the build
META_FILE = "tf_index.json"
# block header: tf, number of documents, bound for ltc cosine, bound for BM25
_BLOCK = struct.Struct("<IIff")
# float32 rounding must never make an upper bound smaller than the true value
_ROUND_UP = 1.0 + 1e-6
# share of all postings that tier 1 and tier 1+2 should hold at most (used to pick the tf thresholds)
TIER_SHARES = (0.10, 0.35)


# one tf block of a term: every document in which the term occurs exactly `tf` times
class Block:
    # plain attribute holder; __slots__ keeps the many small objects cheap
    __slots__ = ("tf", "docs", "bound_ltc", "bound_bm25")

    # store the four fields
    def __init__(self, tf: int, docs: "array.array", bound_ltc: float, bound_bm25: float):
        # tf shared by every block documents
        self.tf = tf
        # doc_ids in increasing order (uint32)
        self.docs = docs
        # max {(1 + log tf) / ||d||} over this block and every later (smaller tf) block of the term
        self.bound_ltc = bound_ltc
        # max {BM25 tf} over this block and every later block of the term
        self.bound_bm25 = bound_bm25


# Impact-ordered postings without positions: for every term the documents are grouped by tf, highest tf first.
#   tf_index.bin   per term: uint32 nblocks, nblocks x (tf, count, bound_ltc, bound_bm25), then the doc_ids (uint32)
#   tf_index.dir   uint64 byte offset of every term's record, in dictionary order
#   tf_index.json  postings size it was built from, tier thresholds, BM25 parameters of the bounds
# The same file serves --impact (blocks in decreasing tf + max-score bounds) and --tiered (tiers = tf ranges).
class TfIndex:

    # open the three files of an existing tf index (use TfIndex.open to build it when missing)
    def __init__(self, idx):
        # index access (term order, N)
        self.idx = idx

        # description written by build()
        with open(os.path.join(idx.index_dir, META_FILE), encoding="utf-8") as f:
            # parsed tf_index.json
            self.meta: dict = json.load(f)

        # lower tf limit of every tier, e.g. [4, 2, 1]
        self.tier_min_tf: List[int] = self.meta["tier_min_tf"]
        # byte offset of every term's record
        self._offsets = array.array("Q")
        # read the whole directory file
        with open(os.path.join(idx.index_dir, DIR_FILE), "rb") as f:
            # uint64 offsets in dictionary order
            self._offsets.frombytes(f.read())

        # the file was written little-endian
        if sys.byteorder != "little":
            # convert on big-endian machines
            self._offsets.byteswap()

        # term -> pos in dict order
        self._term_id: Dict[str, int] = {t: i for i, t in enumerate(idx.all_terms())}
        # one handle for all reads
        self._fp = open(os.path.join(idx.index_dir, BIN_FILE), "rb")

    # True when the three files exist and describe the postings file that is on disk
    @staticmethod
    def available(idx) -> bool:
        # the three file paths
        paths = [os.path.join(idx.index_dir, n) for n in (BIN_FILE, DIR_FILE, META_FILE)]
        # any of them missing
        if not all(os.path.exists(p) for p in paths):
            # needs a build
            return False
        # read tf_index.json
        with open(paths[2], encoding="utf-8") as f:
            # built from a postings file of the same size as the current one
            return json.load(f).get("postings_bytes") == idx.meta["postings_bytes"]

    # open the tf index, building it first when it is missing or stale (one time, about 20 seconds)
    @classmethod
    def open(cls, idx, log=sys.stderr) -> "TfIndex":
        # missing or built from another postings file
        if not cls.available(idx):
            # tell the user why this run is slower
            print("# building tf_index.bin from postings.txt (one time, about 20 seconds) ...", file=log, flush=True)
            # build the three files
            build(idx.index_dir)
        # open them
        return cls(idx)

    # release the file handle
    def close(self) -> None:
        # close tf_index.bin
        self._fp.close()

    # number of tiers
    @property
    def num_tiers(self) -> int:
        # one tier per lower tf limit
        return len(self.tier_min_tf)

    # tier (0 = best) of a tf value
    def tier_of(self, tf: int) -> int:
        # limits are strictly decreasing, so the first one reached is the tier
        for k, lo in enumerate(self.tier_min_tf):
            # tf high enough for tier k
            if tf >= lo:
                # tier number
                return k
        # tf below every limit (cannot happen when the last limit is 1): bottom tier
        return len(self.tier_min_tf) - 1

    # blocks of a term in decreasing tf ([] for a term not in the index)
    def blocks(self, term: str) -> List[Block]:
        # position of the term in dictionary order
        i = self._term_id.get(term)
        # unknown term
        if i is None:
            # no blocks
            return []
        # jump to the term's record
        self._fp.seek(self._offsets[i])
        # number of blocks of the term
        (nblocks,) = struct.unpack("<I", self._fp.read(4))
        # all block headers at once
        head = self._fp.read(nblocks * _BLOCK.size)
        # blocks read so far
        out: List[Block] = []
        # every block, in stored (decreasing tf) order
        for k in range(nblocks):
            # header fields of block k
            tf, count, b_ltc, b_bm25 = _BLOCK.unpack_from(head, k * _BLOCK.size)
            # doc_ids of the block
            docs = array.array("I")
            # the doc_ids follow the headers, block after block
            docs.frombytes(self._fp.read(4 * count))
            # the file was written little-endian
            if sys.byteorder != "little":
                # convert on big-endian machines
                docs.byteswap()
            # keep the block
            out.append(Block(tf, docs, b_ltc, b_bm25))
        # blocks in decreasing tf
        return out

    # (doc_id, tf) of a term; with `allowed` only documents of that set (a small set is looked up by bisect)
    def doc_tfs(self, term: str, allowed: Optional[Set[int]] = None) -> Iterator[Tuple[int, int]]:
        # every block of the term
        for blk in self.blocks(term):
            # its pairs, restricted if asked
            yield from block_pairs(blk, allowed)

    # doc_ids of a term in one tier
    def tier_docs(self, term: str, tier: int, blocks: Optional[List[Block]] = None) -> List["array.array"]:
        # doc arrays of the blocks whose tf falls in the tier (blocks may be passed in to avoid a second read)
        return [b.docs for b in (blocks if blocks is not None else self.blocks(term)) if self.tier_of(b.tf) == tier]


# (doc_id, tf) pairs of one block, optionally restricted to a set of documents
def block_pairs(blk: Block, allowed: Optional[Set[int]] = None) -> Iterator[Tuple[int, int]]:
    # documents and their shared tf
    docs, tf = blk.docs, blk.tf
    # no restriction: the whole block
    if allowed is None:
        # every document of the block
        for d in docs:
            # pair with the block tf
            yield d, tf
    # few allowed documents compared with the block: binary search each of them (cost |allowed| * log |block|)
    elif len(allowed) * 16 < len(docs):
        # block length
        n = len(docs)
        # every allowed document
        for d in allowed:
            # where it would be in the sorted block
            k = bisect.bisect_left(docs, d)
            # present in the block
            if k < n and docs[k] == d:
                # pair with the block tf
                yield d, tf
    # otherwise walk the block and test membership
    else:
        # every document of the block
        for d in docs:
            # keep allowed ones only
            if d in allowed:
                # pair with the block tf
                yield d, tf


# tf thresholds of the tiers from the tf histogram: the smallest tf whose "tf >= x" share stays within each target
def choose_tiers(tf_hist: Dict[int, int], shares: Sequence[float] = TIER_SHARES) -> List[int]:
    # total number of postings
    total = sum(tf_hist.values())
    # lower tf limit found for each target share
    limits: List[int] = []
    # every target share
    for share in shares:
        # postings with tf >= the current candidate
        seen = 0
        # start above the highest tf (an empty tier)
        lo = max(tf_hist) + 1
        # walk from the highest tf down while the cumulative share still fits
        for tf in sorted(tf_hist, reverse=True):
            # adding this tf would exceed the target
            if (seen + tf_hist[tf]) / total > share:
                # stop here
                break
            # include this tf
            seen += tf_hist[tf]
            # new lower limit
            lo = tf
        # limit for this share
        limits.append(lo)
    # thresholds must be strictly decreasing and end with tf >= 1
    out: List[int] = []
    # candidate limits plus the final "every posting" limit
    for lo in limits + [1]:
        # a limit below 1 makes no sense
        lo = max(1, lo)
        # keep only limits smaller than the previous one
        if not out or lo < out[-1]:
            # accept it
            out.append(lo)
    # strictly decreasing limits ending with 1
    return out


# build the three tf index files from postings.txt, doc_norms.tsv and doc_lengths.tsv; returns tf_index.json's content
def build(index_dir: str) -> dict:
    # the teammate's prep module has the postings reader and the weight formulas
    import prep

    # start timer
    t0 = time.time()
    # statistics written by prep.py
    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        # parsed meta.json
        meta = json.load(f)
    # collection size, log base, average document length
    n, log_base, avgdl = meta["N"], meta["log_base"], meta["avgdl"]
    # BM25 parameters of the bounds: the ones the BM25 champion lists were built with
    bm = meta.get("champions", {}).get("schemes", {}).get("bm25", {})
    # fall back to prep.py's defaults when meta.json has none
    k1, b = bm.get("k1", prep.DEFAULT_BM25_K1), bm.get("b", prep.DEFAULT_BM25_B)
    # 1 / ||d|| by doc_id (0 for a document without terms)
    inv_norm = [0.0] * (n + 1)
    # document vector lengths written by prep.py
    with open(os.path.join(index_dir, "doc_norms.tsv"), encoding="utf-8", newline="\n") as f:
        # one "doc_id<TAB>norm" per line
        for line in f:
            # the two fields
            d_s, v_s = line.rstrip("\n").split("\t")
            # reciprocal of the norm
            inv_norm[int(d_s)] = 1.0 / float(v_s) if float(v_s) > 0.0 else 0.0
    # BM25 denominator part k1 (1 - b + b |d| / avgdl) by doc_id
    bm_k = [k1 * (1.0 - b)] * (n + 1)
    # document lengths written by prep.py
    with open(os.path.join(index_dir, "doc_lengths.tsv"), encoding="utf-8", newline="\n") as f:
        # one "doc_id<TAB>length" per line
        for line in f:
            # the two fields
            d_s, l_s = line.rstrip("\n").split("\t")
            # length-normalized saturation constant of this document
            bm_k[int(d_s)] = k1 * (1.0 - b + b * int(l_s) / avgdl)

    # byte offset of every term's record
    offsets = array.array("Q")
    # tf -> number of postings with that tf
    tf_hist: Dict[int, int] = {}
    # blocks written in total
    num_blocks = 0
    # final path of the binary file
    bin_path = os.path.join(index_dir, BIN_FILE)
    # write to a temporary name first, so an interrupted build never leaves a file that looks complete
    with open(bin_path + ".tmp", "wb") as out:
        # bytes written so far
        pos = 0
        # every record of postings.txt, in dictionary order
        for _, _, df, pairs in prep.iter_postings_records(os.path.join(index_dir, "postings.txt")):
            # tf -> doc_ids (postings are in increasing doc_id, so every group stays sorted)
            groups: Dict[int, List[int]] = {}
            # every (doc_id, tf) of the term, as bytes
            for d_b, tf_b in pairs:
                # tf as int
                tf = int(tf_b)
                # existing group for this tf
                g = groups.get(tf)
                # first document with this tf
                if g is None:
                    # new group
                    groups[tf] = [int(d_b)]
                # group exists
                else:
                    # append the doc_id
                    g.append(int(d_b))
            # block order: highest tf first
            tfs = sorted(groups, reverse=True)
            # per block: the best contribution any of its documents can make, then the suffix maximum
            b_ltc = [(1.0 + prep.log_b(tf, log_base)) * max(inv_norm[d] for d in groups[tf]) for tf in tfs]
            # BM25 tf part is largest for the shortest document of the block
            b_bm = [tf * (k1 + 1.0) / (tf + min(bm_k[d] for d in groups[tf])) for tf in tfs]
            # suffix maximum, from the last block backwards
            for k in range(len(tfs) - 2, -1, -1):
                # ltc bound covers this block and every later one
                b_ltc[k] = max(b_ltc[k], b_ltc[k + 1])
                # BM25 bound likewise
                b_bm[k] = max(b_bm[k], b_bm[k + 1])
            # record starts with the number of blocks
            parts = [struct.pack("<I", len(tfs))]
            # one header per block
            for k, tf in enumerate(tfs):
                # tf, count and the rounded-up bounds
                parts.append(_BLOCK.pack(tf, len(groups[tf]), b_ltc[k] * _ROUND_UP, b_bm[k] * _ROUND_UP))
                # histogram for the tier thresholds
                tf_hist[tf] = tf_hist.get(tf, 0) + len(groups[tf])
            # then the doc_ids of every block, in the same order
            for tf in tfs:
                # uint32 array of the block
                a = array.array("I", groups[tf])
                # always store little-endian
                if sys.byteorder != "little":
                    # convert on big-endian machines
                    a.byteswap()
                # raw bytes
                parts.append(a.tobytes())
            # whole record of the term
            rec = b"".join(parts)
            # where it starts
            offsets.append(pos)
            # write it
            out.write(rec)
            # advance the position
            pos += len(rec)
            # count the blocks
            num_blocks += len(tfs)
    # complete: give the binary file its final name
    os.replace(bin_path + ".tmp", bin_path)
    # always store little-endian
    if sys.byteorder != "little":
        # convert on big-endian machines
        offsets.byteswap()
    # write the directory file
    with open(os.path.join(index_dir, DIR_FILE), "wb") as f:
        # uint64 offsets in dictionary order
        f.write(offsets.tobytes())
    # content of tf_index.json
    info = {
        # the tf index is only valid for this exact postings file
        "postings_bytes": meta["postings_bytes"],
        "num_terms": len(offsets),
        "num_postings": sum(tf_hist.values()),
        "num_blocks": num_blocks,
        # tier k holds postings with tier_min_tf[k] <= tf < tier_min_tf[k-1]
        "tier_min_tf": choose_tiers(tf_hist),
        # parameters the BM25 bounds were computed with
        "bm25": {"k1": k1, "b": b},
        "build_seconds": round(time.time() - t0, 1),
    }
    # share of the postings in every tier, for the documentation
    total = info["num_postings"]
    # lower tf limit of every tier
    lims = info["tier_min_tf"]
    # postings with tf in [lims[k], lims[k-1]) divided by all postings
    info["tier_share"] = [
        round(sum(c for tf, c in tf_hist.items() if tf >= lo and (k == 0 or tf < lims[k - 1])) / total, 4)
        for k, lo in enumerate(lims)]
    # written last, so it always describes the files next to it
    with open(os.path.join(index_dir, META_FILE), "w", encoding="utf-8", newline="\n") as f:
        # readable JSON
        json.dump(info, f, indent=2)
        # trailing newline
        f.write("\n")
    # description of the build
    return info


# demo / manual build: python tf_index.py [index folder]
if __name__ == "__main__":
    # index folder from the command line, default output/ next to this script
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    # build and print the description
    print(json.dumps(build(folder), indent=2))
