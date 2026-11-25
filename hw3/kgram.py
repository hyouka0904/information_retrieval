"""Binary K-gram inverted index over dictionary terms and surface vocabulary forms.

Accelerates general wildcard queries. Maps 0 .. T-1 to stemmed index terms and T .. T+S-1 to unstemmed
surface forms, storing sorted uint32 string ID lists in kgram.bin and offset metadata in kgram.json.
"""

import array
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# file names inside the index folder: sorted id lists of every gram
BIN_FILE = "kgram.bin"
# description and gram table
META_FILE = "kgram.json"

# gram length
K = 3
# marks the start and the end of a string
PAD = "$"


# every k-gram of one string, with boundary markers ("$re", "red", "ed$" for "red")
def grams_of(text: str) -> Set[str]:
    # add the boundary markers
    padded = PAD + text + PAD
    # every window of length K
    return {padded[i:i + K] for i in range(len(padded) - K + 1)}


# the k-grams every match of a wildcard pattern must contain (IIR 3.2.2); empty when the pattern is too short
def pattern_grams(pattern: str) -> Set[str]:
    # anchors at both ends; "*" and "?" break the pattern into literal pieces
    pieces = re.split(r"[*?]", PAD + pattern + PAD)
    # grams required by the pattern
    out: Set[str] = set()
    # every literal piece
    for piece in pieces:
        # every window of length K inside the piece (pieces shorter than K give none)
        for i in range(len(piece) - K + 1):
            # required gram
            out.add(piece[i:i + K])
    # possibly empty set
    return out


# String ids: 0 .. T-1 are index terms in dictionary order, T .. T+S-1 are surface forms in file order.
# read-only access to kgram.bin / kgram.json
class KGramIndex:
    """Read-only binary K-gram index for fast wildcard candidate retrieval.

    Attributes:
        meta (dict): Metadata loaded from kgram.json.
        num_terms (int): Number of stemmed dictionary terms (IDs 0 .. num_terms - 1).
        num_surfaces (int): Number of surface vocabulary entries (IDs num_terms ..).
    """

    # load the gram table and open the id file
    def __init__(self, index_dir: str):
        # description and gram table written by build()
        with open(os.path.join(index_dir, META_FILE), encoding="utf-8") as f:
            # parsed kgram.json
            self.meta: dict = json.load(f)
        # gram -> (offset in uint32 units, number of ids)
        self._grams: Dict[str, List[int]] = self.meta.pop("grams")
        # ids below this are index terms
        self.num_terms: int = self.meta["num_terms"]
        # number of surface forms after them
        self.num_surfaces: int = self.meta["num_surfaces"]
        # one handle for all reads
        self._fp = open(os.path.join(index_dir, BIN_FILE), "rb")

    # True when both files exist and were built for this postings file and this surface vocabulary
    @staticmethod
    def available(idx, num_surfaces: int) -> bool:
        # path of kgram.json
        path = os.path.join(idx.index_dir, META_FILE)
        # either file missing
        if not (os.path.exists(path) and os.path.exists(os.path.join(idx.index_dir, BIN_FILE))):
            # needs a build
            return False
        # read only the beginning of kgram.json
        with open(path, encoding="utf-8") as f:
            # enough for the three validity fields
            head = f.read(400)
        # the three numbers sit at the start of the file, so the big gram table is not parsed just for this check
        m = re.search(r'"postings_bytes": (\d+), "num_terms": (\d+), "num_surfaces": (\d+)', head)
        # all three must match the current index and surface vocabulary
        return bool(m) and (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (
            idx.meta["postings_bytes"], idx.vocab_size, num_surfaces)

    # release the file handle
    def close(self) -> None:
        # close kgram.bin
        self._fp.close()

    # sorted string ids containing one gram
    def _ids(self, gram: str) -> "array.array":
        # uint32 ids
        out = array.array("I")
        # offset and length of the gram's list
        entry = self._grams.get(gram)
        # unknown grams have an empty list
        if entry is not None:
            # jump to the list (offset is in uint32 units)
            self._fp.seek(4 * entry[0])
            # read it
            out.frombytes(self._fp.read(4 * entry[1]))
            # the file was written little-endian
            if sys.byteorder != "little":
                # convert on big-endian machines
                out.byteswap()
        # sorted ids
        return out

    # candidate (term ids, surface ids) for a pattern: strings containing every gram of the pattern.
    # None means the pattern has no usable gram (e.g. "*a*") and the caller must scan instead.
    # The result is a superset of the matches ("red*" also proposes "retired"), so a regex check must follow.
    def candidates(self, pattern: str) -> Optional[Tuple[List[int], List[int]]]:
        # grams every match must contain
        grams = pattern_grams(pattern)
        # nothing to intersect
        if not grams:
            # caller falls back to scanning
            return None
        # a gram nobody has means no string can match
        if any(g not in self._grams for g in grams):
            # no term, no surface form
            return [], []
        # shortest list first keeps the running intersection small
        ordered = sorted(grams, key=lambda g: self._grams[g][1])
        # start from the shortest list
        ids = set(self._ids(ordered[0]))
        # intersect with the others
        for g in ordered[1:]:
            # nothing left to intersect
            if not ids:
                # stop early
                break
            # keep ids present in this gram's list too
            ids.intersection_update(self._ids(g))
        # boundary between term ids and surface ids
        t = self.num_terms
        # term ids, and surface ids renumbered from 0
        return sorted(i for i in ids if i < t), sorted(i - t for i in ids if i >= t)


# build kgram.bin / kgram.json from the index terms and the surface forms; returns the description (without the table)
def build(idx, surfaces: List[str]) -> dict:
    # start timer
    t0 = time.time()
    # gram -> ids; ids arrive in increasing order, so every list is sorted
    lists: Dict[str, "array.array"] = {}
    # next string id
    sid = 0
    # index terms first, then surface forms
    for group in (idx.all_terms(), surfaces):
        # every string of the group
        for s in group:
            # every distinct gram of the string
            for g in grams_of(s):
                # the gram's list so far
                a = lists.get(g)
                # first string with this gram
                if a is None:
                    # new list
                    a = lists[g] = array.array("I")
                # add the string id
                a.append(sid)
            # next id
            sid += 1
    # gram -> [offset, length] in uint32 units
    table: Dict[str, List[int]] = {}
    # uint32 values written so far
    pos = 0
    # final path of the id file
    path = os.path.join(idx.index_dir, BIN_FILE)
    # write to a temporary name first
    with open(path + ".tmp", "wb") as f:
        # grams in sorted order
        for g in sorted(lists):
            # the gram's ids
            a = lists[g]
            # where its list starts and how long it is
            table[g] = [pos, len(a)]
            # advance
            pos += len(a)
            # always store little-endian
            if sys.byteorder != "little":
                # convert on big-endian machines
                a.byteswap()
            # raw bytes
            f.write(a.tobytes())
    # complete: give it its final name
    os.replace(path + ".tmp", path)
    # the three validity fields come first (KGramIndex.available reads only the head of the file)
    info = {"postings_bytes": idx.meta["postings_bytes"], "num_terms": idx.vocab_size, "num_surfaces": len(surfaces),
            "k": K, "num_grams": len(table), "num_entries": pos, "build_seconds": round(time.time() - t0, 1)}
    # LF-only JSON
    with open(os.path.join(idx.index_dir, META_FILE), "w", encoding="utf-8", newline="\n") as f:
        # description followed by the gram table, ASCII-escaped
        json.dump({**info, "grams": table}, f, ensure_ascii=True)
    # description without the table
    return info
