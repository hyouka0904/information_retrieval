import argparse
import array
import math
import os
import random
import sys
import time
from typing import Dict, List

# index access and HW1's postings parser
from index_reader import IndexReader, hw1
# surface vocabulary, part of the k-gram index
from spell import SurfaceVocab
# cluster pruning builder
import clusters
# k-gram index builder
import kgram
# tf-ordered postings builder
import tf_index
# file name and SMART letters shared with the weighting module
from weighting import DOC_STATS_FILE, IDF_LETTERS, TF_LETTERS

# text shown by --help
HELP_TEXT = """HW2 second preprocessing step: the extra files of the bonus features. Needs prep.py's output, not the WARC.

  python prep_extra.py                 build whatever is missing or stale
  python prep_extra.py --force         rebuild everything
  python prep_extra.py --only kgram    build one part: tf_index, doc_stats, clusters or kgram

  tf_index.bin/.dir/.json   postings grouped by tf (impact order), for --impact, --tiered, --tf-index
  doc_stats.tsv             per document: length, distinct terms, max tf, vector length for every SMART tf x idf
  cluster_*.tsv             cluster pruning: sqrt(N) leaders and every document's nearest leaders
  kgram.bin/.json           3-gram index over index terms and surface forms, for general wildcards"""
# index folder next to this script
_DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# parts that can be built
PARTS = ("tf_index", "doc_stats", "clusters", "kgram")
# names of the vector-length columns of doc_stats.tsv: norm_<tf letter><idf letter>
NORM_COLUMNS = [f"norm_{t}{i}" for t in TF_LETTERS for i in IDF_LETTERS]


# idf of a term under the three SMART idf letters, in IDF_LETTERS order (n, t, p)
def idf_triple(n: int, df: int, log_base: float) -> List[float]:
    # none, log(N/df), probabilistic idf clipped at 0
    return [1.0, math.log(n / df, log_base), max(0.0, math.log((n - df) / df, log_base)) if df < n else 0.0]


# doc_stats.tsv: length, distinct terms, max tf and ||d|| for all 15 tf x idf combinations, from the tf index
def build_doc_stats(idx: IndexReader, tfi: "tf_index.TfIndex") -> int:
    # collection size and log base
    n, base = idx.N, idx.log_base
    # zero-filled buffer for the double arrays
    zeros = bytes(8 * (n + 1))
    # |d| by doc_id
    length = array.array("d", zeros)
    # number of distinct terms by doc_id
    distinct = array.array("d", zeros)
    # max tf by doc_id
    max_tf = array.array("d", zeros)
    # pass 1: the statistics the "a" and "L" tf variants need
    for term in idx.all_terms():
        # every tf block of the term
        for blk in tfi.blocks(term):
            # tf shared by the block
            tf = blk.tf
            # every document of the block
            for d in blk.docs:
                # add to the document length
                length[d] += tf
                # one more distinct term
                distinct[d] += 1.0
                # new maximum
                if max_tf[d] < tf:
                    # remember it
                    max_tf[d] = tf
    # 1 + log(average tf) of every document, the denominator of "L"
    l_den = [1.0 + math.log(length[d] / distinct[d], base) if distinct[d] else 1.0 for d in range(n + 1)]
    # pass 2: sum of squared weights, one column per tf letter and idf letter
    sq = {t: [array.array("d", zeros) for _ in IDF_LETTERS] for t in TF_LETTERS}
    # local names for the 15 columns (tf letter, idf letter)
    (nn, nt, np_), (ln, lt, lp), (an, at, ap), (bn, bt, bp), (Ln, Lt, Lp) = (sq[t] for t in TF_LETTERS)
    # every term
    for term in idx.all_terms():
        # squared idf under n, t and p
        i0, i1, i2 = (v * v for v in idf_triple(n, idx.df(term), base))
        # every tf block of the term
        for blk in tfi.blocks(term):
            # tf shared by the block
            tf = blk.tf
            # logarithmic tf
            ltf = 1.0 + math.log(tf, base)
            # tf parts that do not depend on the document, squared
            n2, l2 = float(tf * tf), ltf * ltf
            # every document of the block
            for d in blk.docs:
                # augmented tf of this document
                a = 0.5 + 0.5 * tf / max_tf[d]
                # squared
                a2 = a * a
                # log tf relative to the document's average tf
                big = ltf / l_den[d]
                # squared
                big2 = big * big
                # natural tf under the three idfs
                nn[d] += n2 * i0; nt[d] += n2 * i1; np_[d] += n2 * i2
                # logarithmic tf
                ln[d] += l2 * i0; lt[d] += l2 * i1; lp[d] += l2 * i2
                # augmented tf
                an[d] += a2 * i0; at[d] += a2 * i1; ap[d] += a2 * i2
                # boolean tf (weight 1)
                bn[d] += i0; bt[d] += i1; bp[d] += i2
                # log-average tf
                Ln[d] += big2 * i0; Lt[d] += big2 * i1; Lp[d] += big2 * i2
    # columns in NORM_COLUMNS order
    columns = [sq[t][k] for t in TF_LETTERS for k in range(len(IDF_LETTERS))]
    # rows written
    rows = 0
    # LF-only TSV
    with open(os.path.join(idx.index_dir, DOC_STATS_FILE), "w", encoding="utf-8", newline="\n") as f:
        # header line
        f.write("\t".join(["doc_id", "length", "distinct", "max_tf"] + NORM_COLUMNS) + "\n")
        # every doc_id
        for d in range(1, n + 1):
            # documents without terms get no row
            if not distinct[d]:
                # next document
                continue
            # integer statistics
            cells = [str(d), str(int(length[d])), str(int(distinct[d])), str(int(max_tf[d]))]
            # vector lengths with every float digit
            cells += [repr(math.sqrt(c[d])) for c in columns]
            # one row
            f.write("\t".join(cells) + "\n")
            # count it
            rows += 1
    # number of documents written
    return rows


# recompute every column of doc_stats.tsv for sampled documents straight from postings.txt with HW1's parser
def verify_doc_stats(idx: IndexReader, num_docs: int, rng: random.Random) -> float:
    # collection size and log base
    n, base = idx.N, idx.log_base
    # doc_id -> its row of doc_stats.tsv
    stored: Dict[int, List[str]] = {}
    # LF-only line splitting, matching how the file was written
    with open(os.path.join(idx.index_dir, DOC_STATS_FILE), encoding="utf-8", newline="\n") as f:
        # column names
        header = f.readline().rstrip("\n").split("\t")
        # one document per line
        for line in f:
            # its fields
            cells = line.rstrip("\n").split("\t")
            # keyed by doc_id
            stored[int(cells[0])] = cells
    # random documents plus the first and the last one
    sample = set(rng.sample(sorted(stored), min(num_docs, len(stored)))) | {min(stored), max(stored)}
    # doc_id -> [(tf, df)] of every term of the document
    vectors: Dict[int, List] = {d: [] for d in sample}
    # full pass over postings.txt in binary mode
    with open(idx.postings_path, "rb") as fp:
        # one record per line
        for raw in fp:
            # skip blank lines
            if not raw.strip():
                # next record
                continue
            # HW1's own parser
            _, df, plist = hw1.parse_posting_line(raw.decode("utf-8"), attach_skips=False)
            # every posting of the term
            for p in plist:
                # sampled documents only
                if p.doc_id in vectors:
                    # remember tf and the term's df
                    vectors[p.doc_id].append((p.tf, df))
    # largest relative error seen
    worst = 0.0
    # every sampled document
    for d in sorted(sample):
        # its (tf, df) pairs
        vec = vectors[d]
        # tf values
        tfs = [tf for tf, _ in vec]
        # direct length, distinct terms, max tf
        length, distinct, mx = sum(tfs), len(tfs), max(tfs)
        # the three integer columns must match exactly
        assert [int(stored[d][1]), int(stored[d][2]), int(stored[d][3])] == [length, distinct, mx], f"doc {d}: stats"
        # average tf
        avg = length / distinct
        # every vector-length column
        for name in NORM_COLUMNS:
            # tf letter and idf letter from the column name
            t_letter, i_letter = name[-2], name[-1]
            # sum of squared weights
            total = 0.0
            # every term of the document
            for tf, df in vec:
                # tf weight by the SMART definition
                tfw = {"n": tf, "l": 1 + math.log(tf, base), "a": 0.5 + 0.5 * tf / mx, "b": 1.0,
                       "L": (1 + math.log(tf, base)) / (1 + math.log(avg, base))}[t_letter]
                # idf by the SMART definition
                idf = idf_triple(n, df, base)[IDF_LETTERS.index(i_letter)]
                # add the squared weight
                total += (tfw * idf) ** 2
            # direct vector length
            direct = math.sqrt(total)
            # stored value
            got = float(stored[d][header.index(name)])
            # relative error (absolute when the length is 0)
            rel = abs(direct - got) / direct if direct > 0 else abs(got)
            # track the worst
            worst = max(worst, rel)
            # must agree to rounding error
            assert rel < 1e-9, f"doc {d} {name}: {got} vs direct {direct}"
        # the Slide 4 column must agree with the teammate's doc_norms.tsv
        lt = float(stored[d][header.index("norm_lt")])
        # relative tolerance 1e-9
        assert abs(lt - idx.doc_norms[d]) <= 1e-9 * idx.doc_norms[d], f"doc {d}: norm_lt differs from doc_norms.tsv"
    # worst relative error
    return worst


# the tf index must hold exactly the (doc, tf) pairs of postings.txt for sampled terms, and valid bounds
def verify_tf_index(idx: IndexReader, tfi: "tf_index.TfIndex", num_terms: int, rng: random.Random) -> int:
    # every index term in dictionary order
    terms = idx.all_terms()
    # random term positions
    sample = set(rng.sample(range(len(terms)), min(num_terms, len(terms))))
    # the longest lists are the interesting ones
    sample |= set(sorted(range(len(terms)), key=lambda i: idx.df(terms[i]))[-20:])
    # every sampled term
    for i in sample:
        # the term
        term = terms[i]
        # its blocks from the tf index
        blocks = tfi.blocks(term)
        # blocks in strictly decreasing tf
        assert [b.tf for b in blocks] == sorted({b.tf for b in blocks}, reverse=True), f"{term!r}: tf order"
        # doc_ids sorted inside every block
        assert all(list(b.docs) == sorted(b.docs) for b in blocks), f"{term!r}: doc order"
        # same (doc_id, tf) pairs as postings.txt
        assert sorted(tfi.doc_tfs(term)) == idx.get_doc_tfs(term), f"{term!r}: postings differ"
        # every posting's contribution must stay below the bound stored for its block
        for b in blocks:
            # largest (1 + log tf) / ||d|| in the block
            best = max(idx.tf_weight(b.tf) / idx.doc_norms[d] for d in b.docs)
            # stored bound covers it
            assert best <= b.bound_ltc, f"{term!r}: ltc bound too small"
    # number of terms checked
    return len(sample)


# build the requested parts
def run(index_dir: str, only: List[str], force: bool, verify: bool, seed: int = 27) -> None:
    # open the index (restores postings.txt if needed)
    with IndexReader(index_dir) as idx:
        # True when a part was requested
        want = lambda part: part in only
        # tf index first: the other steps read it
        if want("tf_index") and (force or not tf_index.TfIndex.available(idx)):
            # build the three files
            info = tf_index.build(index_dir)
            # report size, tiers and time
            print(f"tf index: {info['num_postings']:,} postings in {info['num_blocks']:,} tf blocks, tiers tf >= "
                  f"{info['tier_min_tf']} hold {info['tier_share']} of the postings ({info['build_seconds']}s)")
        # open it (builds it when it was not requested but is missing)
        tfi = tf_index.TfIndex.open(idx, log=sys.stdout)
        # path of doc_stats.tsv
        stats_path = os.path.join(index_dir, DOC_STATS_FILE)
        # doc_stats.tsv depends only on postings.txt; rebuild it when it is missing (or with --force)
        stale = not os.path.exists(stats_path)
        # doc_stats requested and needed
        if want("doc_stats") and (force or stale):
            # start timer
            t0 = time.time()
            # write the file
            rows = build_doc_stats(idx, tfi)
            # report size and time
            print(f"doc_stats.tsv: {rows:,} documents x {len(NORM_COLUMNS)} vector lengths ({time.time() - t0:.1f}s)")
        # clusters requested and missing or stale
        if want("clusters") and (force or not clusters.Clusters.available(idx)):
            # choose leaders and assign followers
            info = clusters.build(idx, tfi, seed)
            # report size and time
            print(f"cluster pruning: {info['num_leaders']} leaders, every document follows up to {info['max_b1']} "
                  f"({info['build_seconds']}s)")
        # k-gram index requested
        if want("kgram"):
            # surface vocabulary written by prep.py
            vocab = SurfaceVocab.load(index_dir)
            # no surface vocabulary: index terms only
            surfaces = vocab.surfaces if vocab is not None else []
            # missing, stale or forced
            if force or not kgram.KGramIndex.available(idx, len(surfaces)):
                # build kgram.bin / kgram.json
                info = kgram.build(idx, surfaces)
                # report size and time
                print(f"k-gram index: {info['num_grams']:,} {info['k']}-grams, {info['num_entries']:,} entries "
                      f"({info['build_seconds']}s)")
        # verification passes
        if verify:
            # fixed seed, so a rerun samples the same terms and documents
            rng = random.Random(seed)
            # tf index against postings.txt
            print(f"verify: tf index matches postings.txt for {verify_tf_index(idx, tfi, 300, rng)} sampled terms")
            # doc_stats.tsv only if it exists
            if os.path.exists(stats_path):
                # recompute sampled rows from postings.txt
                worst = verify_doc_stats(idx, 20, rng)
                # report the worst relative error
                print(f"verify: doc_stats.tsv matches the direct formulas (worst rel. error {worst:.1e})")
        # release the tf index file
        tfi.close()


# command-line options
def build_arg_parser() -> argparse.ArgumentParser:
    # keep the help text's line breaks
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # index folder
    ap.add_argument("-i", "--index-dir", default=_DEFAULT_INDEX, help="index folder (default: output/)")
    # parts to build
    ap.add_argument("--only", nargs="+", choices=PARTS, default=list(PARTS), help="parts to build (default: all)")
    # ignore the up-to-date checks
    ap.add_argument("--force", action="store_true", help="rebuild even when the files are up to date")
    # skip the verification passes
    ap.add_argument("--no-verify", action="store_true", help="skip the verification passes")
    # parser
    return ap


# run as a script
if __name__ == "__main__":
    # parse the command line
    args = build_arg_parser().parse_args()
    # build the requested parts
    run(args.index_dir, args.only, args.force, not args.no_verify)
