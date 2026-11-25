"""Inject team27's special tokens into our part of ClueWeb09: 07.warc.gz -> 07_team27.warc.gz.

Each entry of special_tokens.json is a token plus an invented fact sentence. The fact is wrapped in <p>...</p> and
inserted right after the <body ...> tag of 2 existing pages (else after </head>, else at the start of the HTTP
payload), and the record's Content-Length is rewritten. Every other byte of the WARC is copied unchanged, so the
document count (31,480) and every other document stay the same.

Target pages come from special_tokens.lock.json when it exists, so every machine injects the same pages. Otherwise
they are chosen with random.Random(seed) among response records whose charset is utf-8 / iso-8859-1 / ascii, that
have a <body> tag and at least 300 body words; no page is used twice. The lock file is then written.

    python inject_tokens.py -i ../07.warc.gz -o ../07_team27.warc.gz
"""
import argparse
import gzip
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import zlib
from typing import Dict, Iterator, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nltk  # noqa: E402

# the bundled stopwords must be found before hw1_indexer is imported
nltk.data.path.insert(0, os.path.join(HERE, "nltk_data"))
import hw1_indexer as hw1  # noqa: E402

TOKENS_PATH = os.path.join(HERE, "special_tokens.json")
LOCK_PATH = os.path.join(HERE, "special_tokens.lock.json")
# codec names (after HW1's alias table) whose bytes an ASCII fact can be inserted into as-is
ALLOWED_CODECS = {"utf-8", "cp1252"}
MIN_BODY_WORDS = 300
DOCS_PER_TOKEN = 2
# doc_text.tsv keeps this many body characters; the fact must fall inside them
BODY_PREFIX = 20000

_BODY_TAG_RE = re.compile(rb"<body\b[^>]*>", re.I)
_HEAD_END_RE = re.compile(rb"</head\s*>", re.I)
_CLEN_RE = re.compile(rb"^(Content-Length\s*:\s*)(\S*)", re.I)


# ----------------------------------------------------------------------
# raw record stream (same record boundaries as hw1.iter_warc_records)
# ----------------------------------------------------------------------

def _open_warc(path: str):
    raw = open(path, "rb")
    is_gzip = raw.read(2) == b"\x1f\x8b"
    raw.seek(0)
    return gzip.GzipFile(fileobj=raw) if is_gzip else raw


def iter_raw(path: str) -> Iterator[Tuple[str, object]]:
    """Yield ("bytes", b"...") for bytes between records and ("record", (header_lines, content)) for each record.
    Concatenating everything back gives the original file. Boundaries follow hw1.iter_warc_records exactly."""
    with _open_warc(path) as f:
        try:
            line = f.readline()
            while line:
                if not line.startswith(b"WARC/"):
                    yield "bytes", line
                    line = f.readline()
                    continue
                lines = [line]
                headers = {}
                while True:
                    line = f.readline()
                    lines.append(line)
                    if not line or line in (b"\r\n", b"\n"):
                        break
                    k, _, v = line.decode("utf-8", "replace").partition(":")
                    headers[k.strip()] = v.strip()
                try:
                    length = max(0, int(headers.get("Content-Length", "0") or 0))
                except ValueError:
                    length = 0
                content = f.read(length)
                yield "record", (lines, headers, content)
                line = f.readline()
        except (EOFError, zlib.error):
            return


def insertion_offset(content: bytes) -> int:
    """Byte offset inside a response record's content where the fact paragraph goes."""
    http_header, payload = hw1.split_http(content)
    start = len(content) - len(payload)
    for rx in (_BODY_TAG_RE, _HEAD_END_RE):
        m = rx.search(payload)
        if m:
            return start + m.end()
    return start


def fact_html(fact: str) -> bytes:
    return f"<p>{html.escape(fact, quote=False)}</p>".encode("ascii")


def _codec(content: bytes) -> str:
    import codecs
    http_header, payload = hw1.split_http(content)
    return codecs.lookup(hw1._detect_charset(http_header, payload)).name


# ----------------------------------------------------------------------
# target selection
# ----------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_candidates(src: str, tokens: List[str], progress: bool = False) -> Tuple[List[int], Dict[int, str], set]:
    """One pass over src: (eligible doc ids, doc_id -> trec_id, words of `tokens` found in some document body).
    A word counts as found when it occurs as a whole word in a lower-cased title or body."""
    words = {w for t in tokens for w in t.lower().split()}
    word_res = {w: re.compile(rf"(?<!\w){re.escape(w)}(?!\w)") for w in words}
    eligible, trec, found = [], {}, set()
    doc_id, t0 = 0, time.time()
    for kind, item in iter_raw(src):
        if kind != "record":
            continue
        lines, headers, content = item
        if headers.get("WARC-Type") != "response":
            continue
        doc_id += 1
        trec[doc_id] = headers.get("WARC-TREC-ID", "")
        doc = _parse_one(content, doc_id, headers)
        text = f"{doc.title} {doc.body}".lower()
        for w, rx in word_res.items():
            if w not in found and rx.search(text):
                found.add(w)
        http_header, payload = hw1.split_http(content)
        if (_codec(content) in ALLOWED_CODECS and _BODY_TAG_RE.search(payload)
                and len(doc.body.split()) >= MIN_BODY_WORDS):
            eligible.append(doc_id)
        if progress and doc_id % 2000 == 0:
            print(f"  scanned {doc_id:,} docs ({time.time() - t0:.0f}s)", file=sys.stderr)
    return eligible, trec, found


def _parse_one(content: bytes, doc_id: int, headers: dict) -> "hw1.Document":
    """hw1.parse_warc's per-record logic, on one record."""
    import codecs
    http_header, payload = hw1.split_http(content)
    charset = hw1._detect_charset(http_header, payload)
    errors = "cp1252_fallback" if codecs.lookup(charset).name == "utf-8" else "replace"
    html_str = payload.decode(charset, errors=errors)
    title, body = hw1.html_to_text(html_str) if html_str.strip() else ("", "")
    return hw1.Document(doc_id, headers.get("WARC-TREC-ID", ""), headers.get("WARC-Target-URI", ""), title, body)


def index_check(tokens: List[str], index_dir: str) -> None:
    """Raise ValueError unless every token has at least one index term with df = 0 in the index at index_dir."""
    meta_path = os.path.join(index_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        icfg = hw1.Config(**json.load(f)["config"])
    df = {}
    need = {t: [term for term, _ in hw1.linguistic(hw1.tokenize(t, icfg), icfg)] for t in tokens}
    wanted = {term for terms in need.values() for term in terms}
    with open(os.path.join(index_dir, "offsets.tsv"), encoding="utf-8", newline="\n") as f:
        for line in f:
            term, df_s, _ = line.rstrip("\n").split("\t")
            if term in wanted:
                df[term] = int(df_s)
    bad = [t for t, terms in need.items() if not terms or all(df.get(term, 0) > 0 for term in terms)]
    if bad:
        raise ValueError(f"token(s) already in the index (every term has df > 0): {', '.join(bad)}; "
                         f"choose rarer tokens")


def choose_targets(entries: List[dict], eligible: List[int], seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    pool = sorted(eligible)
    used: set = set()
    out = []
    for e in entries:
        avail = [d for d in pool if d not in used]
        if len(avail) < DOCS_PER_TOKEN:
            raise ValueError(f"not enough eligible documents for token {e['token']!r}")
        picked = sorted(rng.sample(avail, DOCS_PER_TOKEN))
        used.update(picked)
        out.append(picked)
    return out


# ----------------------------------------------------------------------
# injection
# ----------------------------------------------------------------------

def write_injected(src: str, dst: str, facts_by_doc: Dict[int, List[str]]) -> Dict[int, str]:
    """Copy src to dst (one gzip stream, mtime 0) inserting the facts into the given doc ids.
    Returns doc_id -> trec_id for the targets."""
    tmp = dst + ".tmp"
    trec, doc_id = {}, 0
    with open(tmp, "wb") as raw_out, gzip.GzipFile(fileobj=raw_out, mode="wb", compresslevel=6, mtime=0,
                                                    filename="") as out:
        for kind, item in iter_raw(src):
            if kind == "bytes":
                out.write(item)
                continue
            lines, headers, content = item
            if headers.get("WARC-Type") == "response":
                doc_id += 1
                if doc_id in facts_by_doc:
                    trec[doc_id] = headers.get("WARC-TREC-ID", "")
                    at = insertion_offset(content)
                    content = content[:at] + b"".join(fact_html(f) for f in facts_by_doc[doc_id]) + content[at:]
                    new_lines = []
                    for ln in lines:
                        m = _CLEN_RE.match(ln)
                        if m:
                            ending = ln[len(ln.rstrip(b"\r\n")):]
                            ln = m.group(1) + str(len(content)).encode("ascii") + ending
                        new_lines.append(ln)
                    lines = new_lines
            out.writelines(lines)
            out.write(content)
    missing = set(facts_by_doc) - set(trec)
    if missing:
        os.remove(tmp)
        raise ValueError(f"doc ids not in the WARC: {sorted(missing)}")
    os.replace(tmp, dst)
    return trec


def inject(src: str, dst: str, entries: List[dict], seed: int = 27, lock_path: Optional[str] = LOCK_PATH,
           index_dir: Optional[str] = None, progress: bool = False) -> List[dict]:
    """Inject every entry's fact into 2 documents of src, writing dst. Returns the lock entries
    [{"token", "fact", "doc_ids", "trec_ids"}]. With lock_path pointing to an existing file, its doc ids are reused
    (and its warc_sha256 must match src); otherwise targets are chosen and the lock file is written (if lock_path).
    index_dir: also refuse tokens whose terms all occur in that index (df > 0)."""
    tokens = [e["token"] for e in entries]
    if index_dir is not None:
        index_check(tokens, index_dir)
    src_sha = sha256_file(src)
    lock = None
    if lock_path and os.path.exists(lock_path):
        with open(lock_path, encoding="utf-8") as f:
            lock = json.load(f)
        if lock.get("warc_sha256") != src_sha:
            raise ValueError(f"{os.path.basename(lock_path)} was made from another WARC "
                             f"(sha256 {lock.get('warc_sha256', '?')[:12]}…, input {src_sha[:12]}…)")
        by_token = {e["token"]: e["doc_ids"] for e in lock["entries"]}
        if set(by_token) != set(tokens):
            raise ValueError(f"tokens in {os.path.basename(lock_path)} ({', '.join(by_token)}) differ from the "
                             f"input ({', '.join(tokens)}); delete the lock file to choose new targets")
        targets = [list(by_token[t]) for t in tokens]
    else:
        eligible, _, found = scan_candidates(src, tokens, progress)
        bad = [t for t in tokens if all(w in found for w in t.lower().split())]
        if bad:
            raise ValueError(f"token(s) already in the collection: {', '.join(bad)}; choose rarer tokens")
        targets = choose_targets(entries, eligible, seed)

    facts_by_doc: Dict[int, List[str]] = {}
    for e, docs in zip(entries, targets):
        for d in docs:
            facts_by_doc.setdefault(d, []).append(e["fact"])
    trec = write_injected(src, dst, facts_by_doc)
    result = [{"token": e["token"], "fact": e["fact"], "doc_ids": docs, "trec_ids": [trec[d] for d in docs]}
              for e, docs in zip(entries, targets)]
    if lock_path and lock is None:
        with open(lock_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"warc_sha256": src_sha, "seed": seed, "entries": result}, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-i", "--input", default=os.path.join(os.path.dirname(HERE), "07.warc.gz"))
    ap.add_argument("-o", "--output", default=os.path.join(os.path.dirname(HERE), "07_team27.warc.gz"))
    ap.add_argument("--tokens", default=TOKENS_PATH)
    ap.add_argument("--lock", default=LOCK_PATH)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--check-index", metavar="DIR", help="also refuse tokens already in this index")
    a = ap.parse_args(argv)
    with open(a.tokens, encoding="utf-8") as f:
        entries = json.load(f)
    t0 = time.time()
    reuse = os.path.exists(a.lock)
    print(f"{'reusing ' + a.lock if reuse else 'choosing targets (seed ' + str(a.seed) + ')'}", file=sys.stderr)
    try:
        res = inject(a.input, a.output, entries, a.seed, a.lock, a.check_index, progress=True)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for e in res:
        print(f"{e['token']!r:20} -> docs {e['doc_ids']} {e['trec_ids']}")
    print(f"wrote {a.output} in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
