import argparse
import heapq
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

# make hw1_indexer.py (same folder as this file) importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NLTK data shipped with the project (stopwords), so the program also runs on a machine without internet
import nltk
# searched before the user's own NLTK folders
nltk.data.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data"))
# HW1 code: WARC reader, postings record parser and Config
import hw1_indexer as hw1

# folder that holds this file, hw1_indexer.py and output/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# default index folder (HW1's dictionary.txt / postings.txt live here, new files are written here)
DEFAULT_INDEX_DIR = os.path.join(BASE_DIR, "output")
# default location of the ClueWeb09 WARC file
DEFAULT_WARC = os.path.join(BASE_DIR, "07.warc.gz")
# logarithm base of the Slide 4 formula, shared by tf weight and idf
DEFAULT_LOG_BASE = 10
# number of documents kept per term in a champion list
DEFAULT_CHAMPION_R = 50
# BM25 tf saturation parameter
DEFAULT_BM25_K1 = 1.2
# BM25 length normalization parameter (0 = none, 1 = full)
DEFAULT_BM25_B = 0.75
# champion list orderings that can be built
CHAMPION_SCHEMES = ("cosine", "bm25")
# output file of each champion scheme (ranker.py reads champions.tsv for cosine)
CHAMPION_FILES = {"cosine": "champions.tsv", "bm25": "champions_bm25.tsv"}
# rank boundaries of the tiered index: tier 1 = the best 20 documents of a term, tier 2 = ranks 21-200, tier 3 = the rest
DEFAULT_TIER_SIZES = (20, 200)
# output file of each tier ordering (same orderings as the champion lists)
TIER_FILES = {"cosine": "tiers.tsv", "bm25": "tiers_bm25.tsv"}

# text shown by --help (the full explanation is in README.md)
HELP_TEXT = """HW2 one-time preprocessing: adds what HW1's index lacks for ranking, without rebuilding the index.

  python prep.py                                  everything with the defaults
  python prep.py --champion-schemes cosine        build only the cosine champion lists
  python prep.py --tier-sizes 50 500              tiered index with tiers of ranks 1-50, 51-500 and the rest
  python prep.py --no-warc                        ignore 07.warc.gz (N falls back to the largest doc_id)
  python prep.py --no-surface                     keep the existing surface_vocab.tsv (saves about 4 minutes)"""
# matches the start of every posting "docID, tf: <" inside a record body
PAIR_RE = re.compile(rb"(\d+), (\d+): <")


# logarithm with an explicit base
def log_b(x: float, base: float) -> float:
    # math.log with an explicit base
    return math.log(x, base)


# ----------------------------------------------------------------------
# reading HW1's files
# ----------------------------------------------------------------------

# yield (byte offset, term bytes, df, [(docID bytes, tf bytes), ...]) for every record of postings.txt
def iter_postings_records(postings_path: str) -> Iterator[Tuple[int, bytes, int, List[Tuple[bytes, bytes]]]]:
    # byte offset of the line about to be read
    offset = 0
    # binary mode, so len(line) is the exact byte length on every OS
    with open(postings_path, "rb") as fp:
        # one postings record per line
        for line in fp:
            # this record starts at the current offset
            line_offset = offset
            # advance past this line's raw bytes (including the newline)
            offset += len(line)
            # skip blank lines (HW1's reader tolerates them, so we do too)
            if not line.strip():
                # nothing to index on an empty line
                continue
            # HW1 single-line format must end with '>' (multiline records are not supported)
            assert line.rstrip(b"\r\n").endswith(b">"), f"unexpected record end at byte {line_offset}"
            # split "term, df" from the postings body at the first ": <"
            head, body = line.split(b": <", 1)
            # the term itself may contain ", ", so split from the right like hw1.parse_posting_line
            term_b, df_b = head.rsplit(b", ", 1)
            # document frequency of this term
            df = int(df_b)
            # all (docID, tf) pairs of this term
            pairs = PAIR_RE.findall(body)
            # regex sanity check: exactly one match per posting
            assert len(pairs) == df, f"{term_b!r}: df={df} but parsed {len(pairs)} postings"
            # hand the record to the caller
            yield line_offset, term_b, df, pairs


# [(doc_id, trec_id, url), ...] numbered exactly like hw1.parse_warc (every response record gets an id)
def scan_warc_doc_ids(warc_path: str) -> List[Tuple[int, str, str]]:
    # one entry per response record
    docs: List[Tuple[int, str, str]] = []
    # same counter rule as hw1.parse_warc: starts at 0, +1 for every response record
    doc_id = 0
    # stream records with HW1's fault-tolerant reader (headers only are used)
    for rec in hw1.iter_warc_records(warc_path):
        # warcinfo / request / metadata records do not get a doc_id in HW1
        if rec.headers.get("WARC-Type") != "response":
            # move on to the next record without numbering this one
            continue
        # this response record takes the next doc_id
        doc_id += 1
        # keep the ClueWeb09 id and URL for output and debugging
        docs.append((doc_id, rec.headers.get("WARC-TREC-ID", ""), rec.headers.get("WARC-Target-URI", "")))
    # hand the full list back to the caller
    return docs


# (surface form, index term) -> [cf, df] over the whole collection, using HW1's own pipeline
def scan_surface_vocab(warc_path: str, cfg: "hw1.Config") -> Dict[Tuple[str, str], List[int]]:
    # (surface, term) -> [collection frequency, document frequency]
    vocab: Dict[Tuple[str, str], List[int]] = {}
    # every web page, numbered and parsed exactly like the HW1 build
    for doc in hw1.parse_warc(warc_path):
        # title words first, as in hw1.process_document
        text = f"{doc.title} {doc.body}" if cfg.include_title else doc.body
        # HW1 tokens with their positions
        tokens = hw1.tokenize(text, cfg)
        # position -> surface form as the linguistic module sees it (after case folding only)
        surface_at = {t.pos: (t.text.lower() if cfg.case_folding else t.text) for t in tokens}
        # (surface, term) -> occurrences in this document
        local: Dict[Tuple[str, str], int] = {}
        # every token that survives stopword removal, with the term it became
        for term, pos in hw1.linguistic(tokens, cfg):
            # pair the term with the word it came from
            key = (surface_at[pos], term)
            # count the occurrence
            local[key] = local.get(key, 0) + 1
        # merge into the collection-wide table
        for key, c in local.items():
            # existing entry or a new [0, 0]
            entry = vocab.get(key)
            # first document with this pair
            if entry is None:
                # new counters
                vocab[key] = [c, 1]
            # seen before
            else:
                # add the occurrences
                entry[0] += c
                # one more document
                entry[1] += 1
    # finished table
    return vocab


# write surface_vocab.tsv sorted by surface form, keeping only index terms; returns (lines written, pairs dropped)
def write_surface_vocab(index_dir: str, vocab: Dict[Tuple[str, str], List[int]], index_terms: Set[str]) -> Tuple[int, int]:
    # pairs written and pairs dropped because their term is not in the index
    written, dropped = 0, 0
    # newline="\n" keeps byte-identical files on Windows and Linux
    with open(os.path.join(index_dir, "surface_vocab.tsv"), "w", encoding="utf-8", newline="\n") as f:
        # code-point order, so prefixes form contiguous runs (bisect)
        for (surface, term) in sorted(vocab):
            # a term the index does not have would only lead to empty postings
            if term not in index_terms:
                # count and skip it
                dropped += 1
                # next pair
                continue
            # a tab or newline would break the TSV format
            assert "\t" not in surface and "\n" not in surface and "\r" not in surface, f"bad surface {surface!r}"
            # counters of this pair
            cf, df = vocab[(surface, term)]
            # one line per pair
            f.write(f"{surface}\t{term}\t{cf}\t{df}\n")
            # count it
            written += 1
    # line count and number of dropped pairs
    return written, dropped


# build surface_vocab.tsv, or keep an existing one built for the same postings.txt; returns meta.json's entry or None
def surface_step(index_dir: str, warc_path: Optional[str], scan: "PostingsScan", enabled: bool = True) -> Optional[dict]:
    # build a new file
    if enabled and warc_path and os.path.exists(warc_path):
        # (surface, term) -> [cf, df] with HW1's own pipeline
        vocab = scan_surface_vocab(warc_path, hw1.Config())
        # keep pairs whose term is in the index
        written, dropped = write_surface_vocab(index_dir, vocab, {t for t, _, _ in scan.term_entries})
        # description for meta.json
        return {"file": "surface_vocab.tsv", "pairs": written, "dropped_not_in_index": dropped}
    # previous meta.json and the existing file
    old_meta_path = os.path.join(index_dir, "meta.json")
    # nothing to keep without both of them
    if not (os.path.exists(old_meta_path) and os.path.exists(os.path.join(index_dir, "surface_vocab.tsv"))):
        # no surface vocabulary
        return None
    # previous settings
    with open(old_meta_path, encoding="utf-8") as f:
        # parsed JSON
        old_meta = json.load(f)
    # the file stays valid only for the same postings file (same index terms)
    if old_meta.get("postings_bytes") != scan.postings_bytes:
        # stale file
        return None
    # keep the description of the existing file
    return old_meta.get("surface_vocab")


# ----------------------------------------------------------------------
# pass 1 over postings.txt: offsets, norm sums, document lengths
# ----------------------------------------------------------------------

# everything pass 1 collects from postings.txt
@dataclass
class PostingsScan:
    # (term, df, byte offset) in file order, which is dictionary.txt order
    term_entries: List[Tuple[str, int, int]]
    # per-document S0 = sum (1 + log tf)^2
    s0: Dict[int, float]
    # per-document S1 = sum (1 + log tf)^2 * log df
    s1: Dict[int, float]
    # per-document S2 = sum (1 + log tf)^2 * (log df)^2
    s2: Dict[int, float]
    # per-document |d| = sum of tf over every indexed term
    doc_lengths: Dict[int, int]
    # collection frequency (sum of tf over all documents) of every term, in term_entries order
    term_cf: List[int]
    # total number of (term, doc) postings
    num_postings: int
    # size of postings.txt in bytes
    postings_bytes: int


# pass 1: byte offset of every record, the three norm sums, |d| of every document and cf of every term
def scan_postings(postings_path: str, log_base: float = DEFAULT_LOG_BASE) -> PostingsScan:
    # per-term results, in file order
    term_entries: List[Tuple[str, int, int]] = []
    # per-document running sum S0
    s0: Dict[int, float] = {}
    # per-document running sum S1
    s1: Dict[int, float] = {}
    # per-document running sum S2
    s2: Dict[int, float] = {}
    # per-document running sum of tf
    doc_lengths: Dict[int, int] = {}
    # per-term collection frequency, in file order
    term_cf: List[int] = []
    # cache of (1 + log tf)^2, since the same small tf values repeat millions of times
    tf_weight_sq: Dict[int, float] = {}
    # total number of postings
    num_postings = 0
    # every record of the postings file
    for line_offset, term_b, df, pairs in iter_postings_records(postings_path):
        # remember term, df and where its record starts
        term_entries.append((term_b.decode("utf-8"), df, line_offset))
        # log df is the same for every posting of this term
        ld = log_b(df, log_base)
        # its square, also shared by every posting of this term
        ld2 = ld * ld
        # collection frequency of this term
        cf = 0
        # visit every document containing this term
        for d_b, tf_b in pairs:
            # document id as int
            d = int(d_b)
            # term frequency in that document
            tf = int(tf_b)
            # look up (1 + log tf)^2 in the cache
            a2 = tf_weight_sq.get(tf)
            # first time this tf value is seen
            if a2 is None:
                # compute the squared tf weight once
                a2 = (1.0 + log_b(tf, log_base)) ** 2
                # store it for later postings with the same tf
                tf_weight_sq[tf] = a2
            # accumulate S0 for this document
            s0[d] = s0.get(d, 0.0) + a2
            # accumulate S1 for this document
            s1[d] = s1.get(d, 0.0) + a2 * ld
            # accumulate S2 for this document
            s2[d] = s2.get(d, 0.0) + a2 * ld2
            # accumulate |d| for this document
            doc_lengths[d] = doc_lengths.get(d, 0) + tf
            # accumulate the term's collection frequency
            cf += tf
        # remember the collection frequency
        term_cf.append(cf)
        # count this term's postings
        num_postings += df
    # size of the file actually read
    postings_bytes = os.path.getsize(postings_path)
    # bundle the results
    return PostingsScan(term_entries, s0, s1, s2, doc_lengths, term_cf, num_postings, postings_bytes)


# (N, N_source, max_doc_id): N from the WARC when available, otherwise the largest doc_id
def decide_n(scan: PostingsScan, warc_docs: Optional[List[Tuple[int, str, str]]]) -> Tuple[int, str, int]:
    # largest doc_id that appears in any postings list
    max_doc_id = max(scan.s0)
    # exact N is available from the WARC scan
    if warc_docs is not None:
        # N = number of response records, same as HW1's num_docs
        n = len(warc_docs)
        # postings referencing a doc beyond the WARC means the two files do not match
        assert max_doc_id <= n, f"max doc_id {max_doc_id} > WARC response count {n}"
        # exact value
        return n, "warc", max_doc_id
    # trailing documents without any term are not counted here, so this is a lower bound
    return max_doc_id, "max_doc_id", max_doc_id


# ||d|| = sqrt(L^2 S0 - 2 L S1 + S2) with L = log N (the expanded form of the Slide 4 vector length)
def compute_doc_norms(scan: PostingsScan, n: int, log_base: float = DEFAULT_LOG_BASE) -> Dict[int, float]:
    # log N, the L in the expanded formula
    big_l = log_b(n, log_base)
    # doc_id -> vector length
    norms: Dict[int, float] = {}
    # every document with at least one term
    for d in scan.s0:
        # expanded form of sum_t a^2 (L - l)^2
        sq = big_l * big_l * scan.s0[d] - 2.0 * big_l * scan.s1[d] + scan.s2[d]
        # clamp tiny negative values caused by floating-point cancellation
        norms[d] = math.sqrt(sq) if sq > 0.0 else 0.0
    # finished table
    return norms


# average document length over N documents
def average_doc_length(doc_lengths: Dict[int, int], n: int) -> float:
    # avgdl over all N documents; documents without any term count as length 0
    return sum(doc_lengths.values()) / n


# ----------------------------------------------------------------------
# pass 2 over postings.txt: champion lists
# ----------------------------------------------------------------------

# scheme -> function(doc_id, tf) giving the per-term score that orders a champion list
def champion_scorers(doc_norms: Dict[int, float], doc_lengths: Dict[int, int], n: int, avgdl: float,
                     k1: float, b: float, log_base: float):
    # 1 / ||d|| by doc_id; a zero-length document can never score
    inv_norm = [0.0] * (n + 1)
    # fill every known document
    for d, v in doc_norms.items():
        # reciprocal of the vector length
        inv_norm[d] = 1.0 / v if v > 0.0 else 0.0
    # BM25 denominator part k1 * (1 - b + b * |d| / avgdl) by doc_id
    bm25_k = [k1 * (1.0 - b)] * (n + 1)
    # fill every known document
    for d, length in doc_lengths.items():
        # longer than average documents get a larger denominator
        bm25_k[d] = k1 * (1.0 - b + b * length / avgdl)

    # cosine ordering: (1 + log tf) / ||d||
    def cosine(d: int, tf: int) -> float:
        # document-side weight divided by ||d||; idf is constant within one term, so it is left out
        return (1.0 + log_b(tf, log_base)) * inv_norm[d]

    # BM25 ordering: tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
    def bm25(d: int, tf: int) -> float:
        # BM25 tf part; idf is constant within one term, so it is left out
        return tf * (k1 + 1.0) / (tf + bm25_k[d])

    # one scorer per scheme
    return {"cosine": cosine, "bm25": bm25}


# doc_ids of the r best (doc_id, tf) pairs (higher score first, smaller doc_id on ties), ascending
def top_r_docs(pairs: Sequence[Tuple[int, int]], scorer, r: int) -> List[int]:
    # short lists are kept whole
    if len(pairs) <= r:
        # every document already qualifies
        return [d for d, _ in pairs]
    # r largest by (score, -doc_id)
    best = heapq.nlargest(r, pairs, key=lambda p: (scorer(p[0], p[1]), -p[0]))
    # ascending doc_id, ready for merging
    return sorted(d for d, _ in best)


# pass 2: stream one champion file per scheme; returns scheme -> number of doc_ids written
def write_champions(postings_path: str, index_dir: str, schemes: Sequence[str], r: int,
                    doc_norms: Dict[int, float], doc_lengths: Dict[int, int], n: int, avgdl: float,
                    k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B,
                    log_base: float = DEFAULT_LOG_BASE) -> Dict[str, int]:
    # per-scheme scoring functions
    scorers = champion_scorers(doc_norms, doc_lengths, n, avgdl, k1, b, log_base)
    # one open file per requested scheme; newline="\n" keeps byte-identical files on Windows and Linux
    files = {s: open(os.path.join(index_dir, CHAMPION_FILES[s]), "w", encoding="utf-8", newline="\n")
             for s in schemes}
    # scheme -> number of doc_ids written
    counts = {s: 0 for s in schemes}
    # always close the files
    try:
        # every record of the postings file
        for _, term_b, _, raw_pairs in iter_postings_records(postings_path):
            # term as text
            term = term_b.decode("utf-8")
            # (doc_id, tf) as ints
            pairs = [(int(d), int(tf)) for d, tf in raw_pairs]
            # one line per scheme
            for s in schemes:
                # best r documents of this term under this scheme
                docs = top_r_docs(pairs, scorers[s], r)
                # one "term<TAB>doc doc ..." line; every term is listed so ranker never misses one
                files[s].write(f"{term}\t{' '.join(map(str, docs))}\n")
                # count the ids
                counts[s] += len(docs)
    # cleanup
    finally:
        # every champion file
        for f in files.values():
            # flush and close
            f.close()
    # leftover files of schemes not built this time would describe an older run, so remove them
    for s in CHAMPION_SCHEMES:
        # scheme not requested and an old file exists
        if s not in schemes and os.path.exists(os.path.join(index_dir, CHAMPION_FILES[s])):
            # delete the stale file
            os.remove(os.path.join(index_dir, CHAMPION_FILES[s]))
    # ids written per scheme
    return counts


# ----------------------------------------------------------------------
# pass 3 over postings.txt: tiered index
# ----------------------------------------------------------------------

# doc_ids of each stored tier: tier i holds ranks [sizes[i-1], sizes[i]) of the champion ordering, each ascending
def split_tiers(pairs: Sequence[Tuple[int, int]], scorer, sizes: Sequence[int]) -> List[List[int]]:
    # how many documents the stored tiers hold together
    keep = min(sizes[-1], len(pairs))
    # best keep documents, best first, smaller doc_id on ties (same order as the champion lists)
    ranked = heapq.nsmallest(keep, pairs, key=lambda p: (-scorer(p[0], p[1]), p[0]))
    # rank boundaries of every stored tier
    bounds = [0] + list(sizes)
    # slice the ranking into tiers and sort each one by doc_id
    return [sorted(d for d, _ in ranked[bounds[i]:bounds[i + 1]]) for i in range(len(sizes))]


# pass 3: stream one tiers file per scheme; returns scheme -> doc_ids written per stored tier
def write_tiers(postings_path: str, index_dir: str, schemes: Sequence[str], sizes: Sequence[int],
                doc_norms: Dict[int, float], doc_lengths: Dict[int, int], n: int, avgdl: float,
                k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B,
                log_base: float = DEFAULT_LOG_BASE) -> Dict[str, List[int]]:
    # schemes that actually get a file (none when the tiered index is switched off)
    built = list(schemes) if sizes else []
    # per-scheme scoring functions, the same ones that order the champion lists
    scorers = champion_scorers(doc_norms, doc_lengths, n, avgdl, k1, b, log_base)
    # one open file per scheme; newline="\n" keeps byte-identical files on Windows and Linux
    files = {s: open(os.path.join(index_dir, TIER_FILES[s]), "w", encoding="utf-8", newline="\n") for s in built}
    # scheme -> ids written per stored tier
    counts = {s: [0] * len(sizes) for s in built}
    # always close the files
    try:
        # one pass serves every scheme; skipped entirely when nothing is built
        for _, term_b, _, raw_pairs in (iter_postings_records(postings_path) if built else ()):
            # term as text
            term = term_b.decode("utf-8")
            # (doc_id, tf) as ints
            pairs = [(int(d), int(tf)) for d, tf in raw_pairs]
            # one line per scheme
            for s in built:
                # stored tiers of this term under this scheme
                tiers = split_tiers(pairs, scorers[s], sizes)
                # one "term<TAB>tier-1 docs<TAB>tier-2 docs ..." line; every term is listed, like the champion lists
                files[s].write(term + "\t" + "\t".join(" ".join(map(str, t)) for t in tiers) + "\n")
                # count the ids of every stored tier
                for i, t in enumerate(tiers):
                    # ids in tier i
                    counts[s][i] += len(t)
    # cleanup
    finally:
        # every tiers file
        for f in files.values():
            # flush and close
            f.close()
    # leftover files of schemes not built this time would describe an older run, so remove them
    for s in CHAMPION_SCHEMES:
        # scheme not built and an old file exists
        if s not in built and os.path.exists(os.path.join(index_dir, TIER_FILES[s])):
            # delete the stale file
            os.remove(os.path.join(index_dir, TIER_FILES[s]))
    # ids written per scheme and tier
    return counts


# ----------------------------------------------------------------------
# writing the tables and meta.json
# ----------------------------------------------------------------------

# write offsets.tsv, doc_norms.tsv, doc_lengths.tsv and (with the WARC) doc_ids.tsv
def write_tables(index_dir: str, scan: PostingsScan, doc_norms: Dict[int, float],
                 warc_docs: Optional[List[Tuple[int, str, str]]]) -> None:
    # newline="\n" keeps byte-identical files on Windows and Linux
    with open(os.path.join(index_dir, "offsets.tsv"), "w", encoding="utf-8", newline="\n") as f:
        # same order as dictionary.txt / postings.txt
        for term, df, off in scan.term_entries:
            # a tab or newline inside a term would break the TSV format
            assert "\t" not in term and "\n" not in term, f"bad term {term!r}"
            # one "term<TAB>df<TAB>offset" line
            f.write(f"{term}\t{df}\t{off}\n")
    # newline="\n" keeps byte-identical files on Windows and Linux
    with open(os.path.join(index_dir, "doc_norms.tsv"), "w", encoding="utf-8", newline="\n") as f:
        # ascending doc_id
        for d in sorted(doc_norms):
            # repr-level precision so HW2 reads back exactly the same float
            f.write(f"{d}\t{doc_norms[d]!r}\n")
    # newline="\n" keeps byte-identical files on Windows and Linux
    with open(os.path.join(index_dir, "doc_lengths.tsv"), "w", encoding="utf-8", newline="\n") as f:
        # ascending doc_id
        for d in sorted(scan.doc_lengths):
            # one "doc_id<TAB>length" line
            f.write(f"{d}\t{scan.doc_lengths[d]}\n")
    # the TREC-ID table exists only when the WARC was scanned
    if warc_docs is not None:
        # newline="\n" keeps byte-identical files on Windows and Linux
        with open(os.path.join(index_dir, "doc_ids.tsv"), "w", encoding="utf-8", newline="\n") as f:
            # one line per response record, in doc_id order
            for d, trec_id, url in warc_docs:
                # one "doc_id<TAB>trec_id<TAB>url" line
                f.write(f"{d}\t{trec_id}\t{url}\n")


# content of meta.json: global statistics and the settings every other file was built with
def build_meta(scan: PostingsScan, doc_norms: Dict[int, float], n: int, n_source: str, max_doc_id: int,
               avgdl: float, log_base: float, has_doc_ids: bool, schemes: Sequence[str], r: int,
               k1: float, b: float, surface_info: Optional[dict] = None,
               tier_sizes: Sequence[int] = ()) -> dict:
    # champion settings, one entry per built scheme
    champions = {"r": r, "schemes": {}}
    # tiered index settings, one entry per built scheme (None when it was not built)
    tiers = {"sizes": list(tier_sizes), "schemes": {}} if tier_sizes and schemes else None
    # every scheme that was built
    for s in schemes:
        # file name and ordering rule
        entry = {"file": CHAMPION_FILES[s]}
        # BM25 ordering depends on its parameters
        if s == "bm25":
            # parameters used when the list was ordered
            entry.update({"k1": k1, "b": b})
        # record the scheme
        champions["schemes"][s] = entry
        # the tiers file of the same scheme
        if tiers is not None:
            # same parameters, other file name
            tiers["schemes"][s] = dict(entry, file=TIER_FILES[s])
    # everything HW2 needs besides the tables
    return {
        # number of documents used in idf = log(N / df)
        "N": n,
        # "warc" (exact) or "max_doc_id" (lower bound, no WARC given)
        "N_source": n_source,
        # base shared by tf weight and idf; HW2 query side must use the same one
        "log_base": log_base,
        # vocabulary size
        "num_terms": len(scan.term_entries),
        # total (term, doc) pairs
        "num_postings": scan.num_postings,
        # documents present in doc_norms.tsv
        "num_docs_with_terms": len(doc_norms),
        # largest doc_id in the index
        "max_doc_id": max_doc_id,
        # used by HW2 to detect a stale offsets.tsv after the index is rebuilt
        "postings_bytes": scan.postings_bytes,
        # HW1 default settings, i.e. the configuration of the submitted index
        "config": asdict(hw1.Config()),
        # whether doc_ids.tsv was written
        "has_doc_ids": has_doc_ids,
        # average |d| over N documents, for BM25
        "avgdl": avgdl,
        # sum of |d| over every document
        "total_doc_length": sum(scan.doc_lengths.values()),
        # champion list settings
        "champions": champions,
        # surface vocabulary description, None when it was never built
        "surface_vocab": surface_info,
        # tiered index description, None when it was not built
        "tiers": tiers,
    }


# write meta.json with LF line endings
def write_meta(index_dir: str, meta: dict) -> None:
    # newline="\n" keeps byte-identical files on Windows and Linux
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        # human-readable JSON, non-ASCII kept as is
        json.dump(meta, f, ensure_ascii=False, indent=2)
        # end the file with a newline
        f.write("\n")


# ----------------------------------------------------------------------
# verification (HW1's files and direct formulas are the reference)
# ----------------------------------------------------------------------

# offsets.tsv must list the same (term, df) as dictionary.txt, line by line; returns the line count
def verify_dictionary(dictionary_path: str, scan: PostingsScan) -> int:
    # number of dictionary lines checked
    checked = 0
    # read dictionary.txt as text
    with open(dictionary_path, encoding="utf-8") as f:
        # line i must describe the same term as term_entries[i]
        for i, line in enumerate(f):
            # same right split as HW1, since a term may contain ", "
            term, df_s = line.rstrip("\n").rsplit(", ", 1)
            # dictionary must not be longer than postings
            assert i < len(scan.term_entries), "dictionary.txt has more terms than postings.txt"
            # term and df must both match
            assert (term, int(df_s)) == scan.term_entries[i][:2], f"line {i}: {(term, df_s)} vs {scan.term_entries[i][:2]}"
            # one more line verified
            checked += 1
    # postings must not be longer than dictionary
    assert checked == len(scan.term_entries), "postings.txt has more terms than dictionary.txt"
    # number of matching lines
    return checked


# seek + HW1's parser must return the expected record for sampled terms (plus first and last)
def verify_seeks(postings_path: str, scan: PostingsScan, num_terms: int, rng: random.Random) -> int:
    # random sample of term indices
    sample = set(rng.sample(range(len(scan.term_entries)), min(num_terms, len(scan.term_entries))))
    # always include the first and last term (boundary cases)
    sample |= {0, len(scan.term_entries) - 1}
    # binary mode, matching how offsets were measured
    with open(postings_path, "rb") as fp:
        # ascending order keeps disk access mostly sequential
        for i in sorted(sample):
            # expected term, df and offset
            term, df, off = scan.term_entries[i]
            # jump straight to the record, exactly like HW2 will
            fp.seek(off)
            # parse that one record with HW1's own parser
            p_term, p_df, plist = hw1.parse_posting_line(fp.readline().decode("utf-8"), attach_skips=False)
            # the record at this offset must be the expected term
            assert p_term == term, f"seek {off}: got term {p_term!r}, expected {term!r}"
            # df field and postings count must both match
            assert p_df == df and len(plist) == df, f"{term!r}: df mismatch"
    # number of seeks checked
    return len(sample)


# recompute ||d|| with the direct formula and |d| as a plain sum for sampled documents; returns the worst relative error
def verify_doc_stats(postings_path: str, doc_norms: Dict[int, float], doc_lengths: Dict[int, int], n: int,
                     num_docs: int, rng: random.Random, log_base: float = DEFAULT_LOG_BASE) -> float:
    # random sample of documents
    sample = set(rng.sample(sorted(doc_norms), min(num_docs, len(doc_norms))))
    # always include the smallest and largest doc_id
    sample |= {min(doc_norms), max(doc_norms)}
    # doc_id -> directly accumulated sum of squared weights
    brute_sq = {d: 0.0 for d in sample}
    # doc_id -> directly accumulated tf sum
    brute_len = {d: 0 for d in sample}
    # every record, parsed by HW1's parser this time (independent of PAIR_RE); binary like the other passes
    with open(postings_path, "rb") as fp:
        # one postings record per line
        for raw in fp:
            # decode the record
            line = raw.decode("utf-8")
            # skip blank lines
            if not line.strip():
                # nothing to check on an empty line
                continue
            # term, df and full postings list
            _, df, plist = hw1.parse_posting_line(line, attach_skips=False)
            # idf of this term with the final N
            idf = log_b(n / df, log_base)
            # every posting of this term
            for p in plist:
                # only sampled documents are accumulated
                if p.doc_id in brute_sq:
                    # direct formula from Slide 4, no algebraic expansion
                    w = (1.0 + log_b(p.tf, log_base)) * idf
                    # add the squared weight
                    brute_sq[p.doc_id] += w * w
                    # add the term frequency
                    brute_len[p.doc_id] += p.tf
    # largest relative error of the norms
    worst = 0.0
    # compare every sampled document
    for d in sorted(sample):
        # directly computed vector length
        brute = math.sqrt(brute_sq[d])
        # relative error, guarded for a zero-norm document
        rel = abs(brute - doc_norms[d]) / brute if brute > 0 else abs(doc_norms[d])
        # keep the worst case
        worst = max(worst, rel)
        # lengths are integers, so they must match exactly
        assert brute_len[d] == doc_lengths[d], f"doc {d}: |d| {doc_lengths[d]} vs direct {brute_len[d]}"
    # expanded and direct results must agree to floating-point precision
    assert worst < 1e-9, f"norm mismatch, worst relative error {worst:.2e}"
    # worst error seen
    return worst


# rebuild the champion lists of sampled terms by full sort and compare with the files
def verify_champions(postings_path: str, index_dir: str, scan: PostingsScan, schemes: Sequence[str], r: int,
                     doc_norms: Dict[int, float], doc_lengths: Dict[int, int], n: int, avgdl: float,
                     k1: float, b: float, num_terms: int, rng: random.Random,
                     log_base: float = DEFAULT_LOG_BASE) -> int:
    # random sample of term indices
    sample = set(rng.sample(range(len(scan.term_entries)), min(num_terms, len(scan.term_entries))))
    # the most frequent terms are the interesting case (df > r), so add the 20 largest
    sample |= set(sorted(range(len(scan.term_entries)), key=lambda i: scan.term_entries[i][1])[-20:])
    # sampled term -> its offset
    wanted = {scan.term_entries[i][0]: scan.term_entries[i][2] for i in sample}
    # scheme -> {term: doc_id list} read back from the files
    stored: Dict[str, Dict[str, List[int]]] = {s: {} for s in schemes}
    # every built scheme
    for s in schemes:
        # read the champion file
        with open(os.path.join(index_dir, CHAMPION_FILES[s]), encoding="utf-8", newline="\n") as f:
            # every line, counted
            for i, line in enumerate(f):
                # term and id list
                term, ids = line.rstrip("\n").split("\t", 1)
                # line order must be dictionary order
                assert term == scan.term_entries[i][0], f"{s}: line {i} is {term!r}"
                # keep sampled terms only
                if term in wanted:
                    # parsed ids
                    stored[s][term] = [int(x) for x in ids.split()]
            # every term must have a line
            assert i + 1 == len(scan.term_entries), f"{s}: {i + 1} lines for {len(scan.term_entries)} terms"
    # read sampled records directly with HW1's parser
    with open(postings_path, "rb") as fp:
        # every sampled term
        for term, off in wanted.items():
            # jump to its record
            fp.seek(off)
            # full postings list
            _, _, plist = hw1.parse_posting_line(fp.readline().decode("utf-8"), attach_skips=False)
            # every built scheme
            for s in schemes:
                # direct score of one posting under this scheme
                if s == "cosine":
                    # (1 + log tf) / ||d||
                    score = lambda p: (1.0 + log_b(p.tf, log_base)) / doc_norms[p.doc_id]
                # BM25 tf part
                else:
                    # tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
                    score = lambda p: p.tf * (k1 + 1.0) / (p.tf + k1 * (1.0 - b + b * doc_lengths[p.doc_id] / avgdl))
                # full sort, best first, smaller doc_id on ties
                ranked = sorted(plist, key=lambda p: (-score(p), p.doc_id))
                # expected champion list in ascending doc_id
                expect = sorted(p.doc_id for p in ranked[:r])
                # file content must match
                assert stored[s][term] == expect, f"{s}: champion list of {term!r} differs"
    # number of terms checked per scheme
    return len(wanted)


# rebuild the tiers of sampled terms by full sort and compare with the tiers files; returns terms checked per scheme
def verify_tiers(postings_path: str, index_dir: str, scan: PostingsScan, schemes: Sequence[str],
                 sizes: Sequence[int], doc_norms: Dict[int, float], doc_lengths: Dict[int, int], n: int,
                 avgdl: float, k1: float, b: float, num_terms: int, rng: random.Random,
                 log_base: float = DEFAULT_LOG_BASE) -> int:
    # random sample of term indices
    sample = set(rng.sample(range(len(scan.term_entries)), min(num_terms, len(scan.term_entries))))
    # the most frequent terms are the interesting case (df > the largest size), so add the 20 largest
    sample |= set(sorted(range(len(scan.term_entries)), key=lambda i: scan.term_entries[i][1])[-20:])
    # sampled term -> its offset
    wanted = {scan.term_entries[i][0]: scan.term_entries[i][2] for i in sample}
    # scheme -> {term: list of tiers} read back from the files
    stored: Dict[str, Dict[str, List[List[int]]]] = {s: {} for s in schemes}
    # every built scheme
    for s in schemes:
        # read the tiers file
        with open(os.path.join(index_dir, TIER_FILES[s]), encoding="utf-8", newline="\n") as f:
            # every line, counted
            for i, line in enumerate(f):
                # term followed by one field per stored tier
                fields = line.rstrip("\n").split("\t")
                # line order must be dictionary order
                assert fields[0] == scan.term_entries[i][0], f"{s}: line {i} is {fields[0]!r}"
                # exactly one field per stored tier
                assert len(fields) == len(sizes) + 1, f"{s}: wrong field count for {fields[0]!r}"
                # keep sampled terms only
                if fields[0] in wanted:
                    # parsed ids of every tier
                    stored[s][fields[0]] = [[int(x) for x in fld.split()] for fld in fields[1:]]
            # every term must have a line
            assert i + 1 == len(scan.term_entries), f"{s}: {i + 1} lines for {len(scan.term_entries)} terms"
    # read sampled records directly with HW1's parser
    with open(postings_path, "rb") as fp:
        # every sampled term
        for term, off in wanted.items():
            # jump to its record
            fp.seek(off)
            # full postings list
            _, _, plist = hw1.parse_posting_line(fp.readline().decode("utf-8"), attach_skips=False)
            # every built scheme
            for s in schemes:
                # direct score of one posting under this scheme
                if s == "cosine":
                    # (1 + log tf) / ||d||
                    score = lambda p: (1.0 + log_b(p.tf, log_base)) / doc_norms[p.doc_id]
                # BM25 tf part
                else:
                    # tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
                    score = lambda p: p.tf * (k1 + 1.0) / (p.tf + k1 * (1.0 - b + b * doc_lengths[p.doc_id] / avgdl))
                # full sort, best first, smaller doc_id on ties
                ranked = [p.doc_id for p in sorted(plist, key=lambda p: (-score(p), p.doc_id))]
                # rank boundaries
                bounds = [0] + list(sizes)
                # expected tiers, each in ascending doc_id
                expect = [sorted(ranked[bounds[k]:bounds[k + 1]]) for k in range(len(sizes))]
                # file content must match
                assert stored[s][term] == expect, f"{s}: tiers of {term!r} differ"
    # number of terms checked per scheme
    return len(wanted)


# structure and mapping checks of surface_vocab.tsv, plus how much of the index it covers
def verify_surface_vocab(index_dir: str, scan: PostingsScan, cfg: "hw1.Config") -> Dict[str, float]:
    # every index term, for membership tests
    index_terms = {t for t, _, _ in scan.term_entries}
    # term -> collection frequency in the index
    index_cf = {t: cf for (t, _, _), cf in zip(scan.term_entries, scan.term_cf)}
    # term -> collection frequency summed over its surface forms
    surface_cf: Dict[str, int] = {}
    # previous surface form, to check the sort order
    prev = None
    # number of lines read
    lines = 0
    # LF-only line splitting, matching how the file was written
    with open(os.path.join(index_dir, "surface_vocab.tsv"), encoding="utf-8", newline="\n") as f:
        # one "surface<TAB>term<TAB>cf<TAB>df" per line
        for line in f:
            # split into the four fields
            surface, term, cf_s, df_s = line.rstrip("\n").split("\t")
            # strictly increasing: sorted, and one line per surface form
            assert prev is None or prev < surface, f"surface_vocab.tsv not sorted / duplicated at {surface!r}"
            # remember for the next line
            prev = surface
            # only index terms are written
            assert term in index_terms, f"{surface!r} -> {term!r} is not an index term"
            # the linguistic module must turn this one token into exactly this term
            assert [t for t, _ in hw1.linguistic([hw1.Token(surface, 0)], cfg)] == [term], f"{surface!r} -> {term!r}"
            # counters must be positive and df can never exceed cf
            assert 0 < int(df_s) <= int(cf_s), f"{surface!r}: bad counters"
            # add to the term's total
            surface_cf[term] = surface_cf.get(term, 0) + int(cf_s)
            # count the line
            lines += 1
    # share of index terms that have at least one surface form
    term_cover = len(surface_cf) / len(index_terms)
    # share of index token occurrences accounted for (never counting more than the index has)
    cf_cover = sum(min(c, index_cf[t]) for t, c in surface_cf.items()) / sum(scan.term_cf)
    # summary numbers
    return {"pairs": lines, "index_terms_covered": term_cover, "cf_covered": cf_cover}


# ----------------------------------------------------------------------
# whole pipeline
# ----------------------------------------------------------------------

# run every step in order, then the verification; returns meta.json's content
def run(index_dir: str = DEFAULT_INDEX_DIR, warc_path: Optional[str] = DEFAULT_WARC,
        log_base: float = DEFAULT_LOG_BASE, schemes: Sequence[str] = CHAMPION_SCHEMES,
        r: int = DEFAULT_CHAMPION_R, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B,
        verify: bool = True, seed: int = 27, surface: bool = True,
        tier_sizes: Sequence[int] = DEFAULT_TIER_SIZES) -> dict:
    # HW1 postings file
    postings_path = os.path.join(index_dir, "postings.txt")
    # HW1 dictionary file
    dictionary_path = os.path.join(index_dir, "dictionary.txt")
    # fail early with a clear message if the HW1 index is missing
    for p in (postings_path, dictionary_path):
        # required input
        if not os.path.exists(p):
            # explain what is missing
            raise FileNotFoundError(f"{p} not found; put HW1's dictionary.txt and postings.txt in {index_dir}")
    # tier sizes must be positive and strictly increasing, otherwise some tier would always be empty
    if any(t < 1 for t in tier_sizes) or any(a >= c for a, c in zip(tier_sizes, tier_sizes[1:])):
        # explain the rule
        raise ValueError(f"tier sizes must be positive and strictly increasing, got {list(tier_sizes)}")

    # step 1: N and TREC-IDs from the WARC
    warc_docs = None
    # only scan when the WARC file is present
    if warc_path and os.path.exists(warc_path):
        # start timer
        t0 = time.time()
        # number every response record
        warc_docs = scan_warc_doc_ids(warc_path)
        # report
        print(f"[1/7] WARC: {len(warc_docs):,} response records ({time.time() - t0:.1f}s)")
    # no WARC given
    else:
        # explain the fallback
        print("[1/7] WARC not found, skipped; N falls back to the largest doc_id in postings.txt")

    # step 2: pass 1 over postings
    t0 = time.time()
    # offsets, norm sums, lengths
    scan = scan_postings(postings_path, log_base)
    # report
    print(f"[2/7] postings pass 1: {len(scan.term_entries):,} terms, {scan.num_postings:,} postings "
          f"({time.time() - t0:.1f}s)")

    # step 3: N, norms, avgdl, tables
    n, n_source, max_doc_id = decide_n(scan, warc_docs)
    # vector lengths
    doc_norms = compute_doc_norms(scan, n, log_base)
    # BM25 average document length
    avgdl = average_doc_length(scan.doc_lengths, n)
    # write offsets / norms / lengths / doc_ids
    write_tables(index_dir, scan, doc_norms, warc_docs)
    # report
    print(f"[3/7] N = {n:,} ({n_source}), avgdl = {avgdl:.2f}; tables written")

    # step 4: pass 2, champion lists
    t0 = time.time()
    # one file per scheme
    counts = write_champions(postings_path, index_dir, schemes, r, doc_norms, scan.doc_lengths, n, avgdl,
                             k1, b, log_base)
    # report
    print(f"[4/7] champion lists (r={r}): " + ", ".join(f"{s} {counts[s]:,} ids" for s in schemes)
          + f" ({time.time() - t0:.1f}s)")

    # step 5: pass 3, tiered index
    t0 = time.time()
    # one file per scheme, same orderings as the champion lists
    tier_counts = write_tiers(postings_path, index_dir, schemes, tier_sizes, doc_norms, scan.doc_lengths, n, avgdl,
                              k1, b, log_base)
    # report
    print(f"[5/7] tiered index (sizes {list(tier_sizes)}): "
          + ("; ".join(f"{s} " + ", ".join(f"tier {i + 1} {c:,} ids" for i, c in enumerate(cs))
                       for s, cs in tier_counts.items()) or "skipped")
          + f" ({time.time() - t0:.1f}s)")

    # step 6: surface vocabulary (needs the WARC; the slowest step because every page is parsed again)
    t0 = time.time()
    # build it, or keep the description of an existing file
    surface_info = surface_step(index_dir, warc_path if warc_docs is not None else None, scan, surface)
    # report
    print(f"[6/7] surface vocabulary: " + (f"{surface_info['pairs']:,} surface forms" if surface_info else "none")
          + f" ({time.time() - t0:.1f}s)")

    # step 7: meta.json last, so it always matches the files next to it
    meta = build_meta(scan, doc_norms, n, n_source, max_doc_id, avgdl, log_base, warc_docs is not None,
                      schemes, r, k1, b, surface_info, tier_sizes)
    # write it
    write_meta(index_dir, meta)
    # report
    print("[7/7] meta.json written")

    # optional verification
    if verify:
        # reproducible samples
        rng = random.Random(seed)
        # dictionary agreement
        print(f"verify: {verify_dictionary(dictionary_path, scan):,} terms match dictionary.txt")
        # seek path
        print(f"verify: {verify_seeks(postings_path, scan, 2000, rng):,} seeks returned the right record")
        # norms and lengths
        worst = verify_doc_stats(postings_path, doc_norms, scan.doc_lengths, n, 20, rng, log_base)
        # report
        print(f"verify: norms and lengths of sampled documents match (worst rel. error {worst:.1e})")
        # champion lists
        if schemes:
            # report
            checked = verify_champions(postings_path, index_dir, scan, schemes, r, doc_norms, scan.doc_lengths,
                                       n, avgdl, k1, b, 300, rng, log_base)
            # report
            print(f"verify: champion lists of {checked} sampled terms match a full sort ({', '.join(schemes)})")
        # tiered index, when built
        if tier_sizes and schemes:
            # rebuild sampled tiers by full sort
            checked = verify_tiers(postings_path, index_dir, scan, schemes, tier_sizes, doc_norms, scan.doc_lengths,
                                   n, avgdl, k1, b, 300, rng, log_base)
            # report
            print(f"verify: tiers of {checked} sampled terms match a full sort ({', '.join(schemes)})")
        # surface vocabulary, when present
        if surface_info is not None:
            # structure, mapping and coverage
            info = verify_surface_vocab(index_dir, scan, hw1.Config())
            # report
            print(f"verify: surface vocabulary OK; covers {info['index_terms_covered']:.2%} of index terms and "
                  f"{info['cf_covered']:.2%} of token occurrences")
    # global statistics
    return meta


# command-line options (every option is listed in README.md)
def build_arg_parser() -> argparse.ArgumentParser:
    # parser with the module docstring as help text
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # index folder
    ap.add_argument("-i", "--index-dir", default=DEFAULT_INDEX_DIR, help="folder with HW1's index (default: output/)")
    # WARC location
    ap.add_argument("--warc", default=DEFAULT_WARC, help="ClueWeb09 WARC file (default: 07.warc.gz next to prep.py)")
    # ignore the WARC
    ap.add_argument("--no-warc", action="store_true", help="do not read the WARC even if it exists")
    # log base
    ap.add_argument("--log-base", type=float, default=DEFAULT_LOG_BASE, help="log base of tf weight and idf (default: 10)")
    # champion list / tier orderings
    ap.add_argument("--champion-schemes", nargs="*", choices=CHAMPION_SCHEMES, default=list(CHAMPION_SCHEMES),
                    help="orderings of the champion lists and tiers to build (default: cosine bm25; give none to skip)")
    # champion list length
    ap.add_argument("--champion-r", type=int, default=DEFAULT_CHAMPION_R, help="documents per champion list (default: 50)")
    # BM25 k1
    ap.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1, help="BM25 k1 for the bm25 ordering (default: 1.2)")
    # BM25 b
    ap.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B, help="BM25 b for the bm25 ordering (default: 0.75)")
    # tiered index rank boundaries
    ap.add_argument("--tier-sizes", nargs="*", type=int, default=list(DEFAULT_TIER_SIZES),
                    help="rank boundaries of the tiered index, built for every champion scheme "
                         "(default: 20 200; give none to skip)")
    # skip the slow surface vocabulary step
    ap.add_argument("--no-surface", action="store_true",
                    help="do not rebuild surface_vocab.tsv (an existing one is kept); saves about 4 minutes")
    # skip the checks
    ap.add_argument("--no-verify", action="store_true", help="skip the verification passes")
    # finished parser
    return ap


# run as a script
if __name__ == "__main__":
    # parse command-line options
    args = build_arg_parser().parse_args()
    # integer base keeps meta.json identical to the notebook's ("log_base": 10, not 10.0)
    base = int(args.log_base) if args.log_base == int(args.log_base) else args.log_base
    # whole pipeline
    run(args.index_dir, None if args.no_warc else args.warc, base, args.champion_schemes,
        args.champion_r, args.bm25_k1, args.bm25_b, not args.no_verify, surface=not args.no_surface,
        tier_sizes=args.tier_sizes)
