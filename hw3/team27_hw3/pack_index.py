import argparse
import hashlib
import json
import lzma
import os
import re
import sys
import time
from itertools import accumulate
from typing import Optional

# text shown by --help
HELP_TEXT = """Lossless delta-gap encoder and LZMA (.xz) archiver for HW1's postings.txt.

  python pack_index.py pack       output/postings.txt -> output/postings.delta.xz (+ postings.pack.json), verified
  python pack_index.py unpack     the reverse; search.py does this by itself on its first run

postings.txt (200 MB) only zips down to 63 MB. Stored as gaps (doc_id gaps, position gaps, tf and df implied by the
counts) and compressed with xz it is about 31 MB, and the restored file is byte-identical (size and SHA-256 checked),
so every byte offset in offsets.tsv stays valid."""
# index folder next to this script
_DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# packed postings file
PACKED_FILE = "postings.delta.xz"
# size and hash of the original postings.txt, written by pack()
MANIFEST_FILE = "postings.pack.json"
# one posting of HW1's format: "docID, tf: <pos, pos, ...>"
_POSTING_RE = re.compile(r"(\d+), (\d+): <([^>]*)>")


# "term, df: <doc, tf: <p, p>; doc, tf: <p>>"  ->  "term<TAB>docgap:pgap,pgap;docgap:pgap"
def encode_line(line: str) -> str:
    # split the "term, df" head from the postings body
    head, body = line.split(": <", 1)
    # the term itself may contain ", ", so split from the right like HW1's parser
    term, df = head.rsplit(", ", 1)
    # encoded postings of this term
    parts = []
    # previous doc_id, for doc_id gaps
    prev_doc = 0
    # every posting of the record
    for d_s, tf_s, pos_s in _POSTING_RE.findall(body):
        # doc_id as int
        d = int(d_s)
        # previous position and the position gaps of this posting
        prev, gaps = 0, []
        # every position of the posting
        for p_s in pos_s.split(", "):
            # position as int
            p = int(p_s)
            # store the gap to the previous position
            gaps.append(str(p - prev))
            # move the reference point
            prev = p
        # tf is not stored: it must equal the number of positions
        assert len(gaps) == int(tf_s), f"{term!r}: tf {tf_s} but {len(gaps)} positions in doc {d}"
        # "docgap:pgap,pgap,..."
        parts.append(f"{d - prev_doc}:{','.join(gaps)}")
        # move the doc_id reference point
        prev_doc = d
    # df is not stored either: it must equal the number of postings
    assert len(parts) == int(df), f"{term!r}: df {df} but {len(parts)} postings"
    # one encoded line per term
    return f"{term}\t{';'.join(parts)}"


# the exact inverse of encode_line
def decode_line(line: str) -> str:
    # split the term from the encoded postings
    term, body = line.split("\t", 1)
    # decoded postings in HW1's format
    out = []
    # running doc_id
    d = 0
    # one entry per posting
    posts = body.split(";")
    # every encoded posting
    for post in posts:
        # doc_id gap and the position gaps
        gap, pos_s = post.split(":", 1)
        # restore the doc_id
        d += int(gap)
        # most postings have a single position: no running sum needed
        if "," not in pos_s:
            # tf = 1, the gap is the position itself
            out.append(f"{d}, 1: <{pos_s}>")
        # several positions
        else:
            # running sum of the gaps gives the positions
            pos = list(accumulate(map(int, pos_s.split(","))))
            # tf = number of positions
            out.append(f"{d}, {len(pos)}: <{', '.join(map(str, pos))}>")
    # df = number of postings, written exactly as HW1 does
    return f"{term}, {len(posts)}: <{'; '.join(out)}>"


# SHA-256 of a file, read in 1 MiB chunks
def _sha256(path: str) -> str:
    # running hash
    h = hashlib.sha256()
    # binary mode: hash the exact bytes
    with open(path, "rb") as f:
        # read until EOF
        for chunk in iter(lambda: f.read(1 << 20), b""):
            # add the chunk
            h.update(chunk)
    # hex digest
    return h.hexdigest()


# postings.txt -> postings.delta.xz + manifest; the packed file is restored once to prove it is lossless
def pack(index_dir: str, preset: int = 9) -> dict:
    # original postings file
    src = os.path.join(index_dir, "postings.txt")
    # packed output
    dst = os.path.join(index_dir, PACKED_FILE)
    # start timer
    t0 = time.time()
    # records written
    lines = 0
    # newline="\n": only LF ends a record (terms may contain other Unicode line separators)
    with open(src, encoding="utf-8", newline="\n") as fin, \
            lzma.open(dst, "wt", encoding="utf-8", newline="\n", preset=preset) as fout:
        # one record per line
        for line in fin:
            # every HW1 record ends with ">" and a newline
            assert line.endswith(">\n"), "postings.txt: unexpected record end"
            # encode without the newline, then add it back
            fout.write(encode_line(line[:-1]) + "\n")
            # count the record
            lines += 1
    # manifest: what the restored file must look like
    info = {"format": 1, "postings_bytes": os.path.getsize(src), "sha256": _sha256(src), "records": lines,
            "packed_bytes": os.path.getsize(dst)}
    # LF-only, same as every other output file
    with open(os.path.join(index_dir, MANIFEST_FILE), "w", encoding="utf-8", newline="\n") as f:
        # readable JSON
        json.dump(info, f, indent=2)
        # trailing newline
        f.write("\n")
    # round trip into a scratch file: unpack() raises unless size and hash match the original
    check = src + ".check"
    # restore and verify
    unpack(index_dir, target=check, log=None)
    # the check copy is not needed any more
    os.remove(check)
    # total time, reported to the caller only (not stored in the manifest)
    info["pack_seconds"] = round(time.time() - t0, 1)
    # manifest plus timing
    return info


# postings.delta.xz -> postings.txt (or `target`), verified against the manifest before it gets its final name
def unpack(index_dir: str, target: Optional[str] = None, log=sys.stderr) -> str:
    # default target is the real postings.txt
    target = target or os.path.join(index_dir, "postings.txt")
    # read the manifest
    with open(os.path.join(index_dir, MANIFEST_FILE), encoding="utf-8") as f:
        # expected size and hash
        info = json.load(f)
    # start timer
    t0 = time.time()
    # write to a temporary name first
    tmp = target + ".tmp"
    # running hash of the restored bytes
    h = hashlib.sha256()
    # restored bytes so far
    size = 0
    # decode line by line into the temporary file
    with lzma.open(os.path.join(index_dir, PACKED_FILE), "rt", encoding="utf-8", newline="\n") as fin, \
            open(tmp, "wb") as fout:
        # decoded records waiting to be written
        buf = []
        # every encoded record, counted from 1
        for n, line in enumerate(fin, start=1):
            # decode without the newline
            buf.append(decode_line(line[:-1]))
            # write in batches of records
            if len(buf) >= 2000:
                # batch as bytes, each record followed by LF
                data = ("\n".join(buf) + "\n").encode("utf-8")
                # start a new batch
                buf.clear()
                # hash the batch
                h.update(data)
                # count its bytes
                size += len(data)
                # write it
                fout.write(data)
                # progress roughly every 50,000 records
                if log is not None and n % 50000 < 2000:
                    # overwrite the same line
                    print(f"\r# restoring postings.txt: {size / info['postings_bytes']:4.0%}", end="", file=log, flush=True)
        # last, partial batch
        if buf:
            # batch as bytes
            data = ("\n".join(buf) + "\n").encode("utf-8")
            # hash it
            h.update(data)
            # count it
            size += len(data)
            # write it
            fout.write(data)
    # the restored file must be byte-identical to the packed original
    if size != info["postings_bytes"] or h.hexdigest() != info["sha256"]:
        # never leave a broken file behind
        os.remove(tmp)
        # stop loudly
        raise RuntimeError(f"{PACKED_FILE} did not restore to the original postings.txt (size or SHA-256 mismatch)")
    # verified: give it its final name
    os.replace(tmp, target)
    # final progress line
    if log is not None:
        # size, verification and duration
        print(f"\r# restored postings.txt: {size:,} bytes, SHA-256 verified ({time.time() - t0:.0f}s)", file=log)
    # path of the restored file
    return target


# called by IndexReader and prep.py: restore postings.txt when only the packed form was shipped
def ensure_postings(index_dir: str, log=sys.stderr) -> None:
    # nothing to do when postings.txt is already there
    if os.path.exists(os.path.join(index_dir, "postings.txt")):
        # keep it
        return
    # restore only when both the packed file and its manifest exist
    if os.path.exists(os.path.join(index_dir, PACKED_FILE)) and os.path.exists(os.path.join(index_dir, MANIFEST_FILE)):
        # tell the user why the first run is slower
        print(f"# first run: restoring postings.txt from {PACKED_FILE} (one time, about 15 seconds) ...", file=log, flush=True)
        # restore and verify
        unpack(index_dir, log=log)


# run as a script
if __name__ == "__main__":
    # keep the help text's line breaks
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # pack or unpack
    ap.add_argument("action", choices=("pack", "unpack"))
    # index folder
    ap.add_argument("-i", "--index-dir", default=_DEFAULT_INDEX, help="index folder (default: output/)")
    # parse the command line
    args = ap.parse_args()
    # pack
    if args.action == "pack":
        # pack and print the manifest
        print(json.dumps(pack(args.index_dir), indent=2))
    # unpack
    else:
        # restore postings.txt
        unpack(args.index_dir)
