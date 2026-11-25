"""Rebuild the HW1 index from our WARC with team27's special tokens injected (runs on Windows and Linux).

Steps (each a subprocess, stopped at the first failure):
  inject    python inject_tokens.py -i ../07.warc.gz -o ../07_team27.warc.gz   (reuses special_tokens.lock.json)
  index     python hw1_indexer.py -i ../07_team27.warc.gz -o output           (~4-5 min, HW2's Config)
  prep      python prep.py -i output --warc ../07_team27.warc.gz              (~4-5 min)
  extra     python prep_extra.py -i output --force
  doc_text  python -m rag.doc_store build --warc ../07_team27.warc.gz
  check     postings SHA-256 vs canonical_index.json (warning on mismatch; --canonical writes it instead), and
            every lock entry's 2 docs must rank first for its token (exit 1 otherwise)

    python rebuild_index.py                     # everything
    python rebuild_index.py --from doc_text     # resume at a step
    python rebuild_index.py --canonical         # the canonical (Windows) build: write canonical_index.json
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
STEPS = ["inject", "index", "prep", "extra", "doc_text", "check"]
LOCK_PATH = os.path.join(HERE, "special_tokens.lock.json")
CANONICAL_PATH = os.path.join(HERE, "canonical_index.json")
NLTK_DIR = os.path.join(HERE, "nltk_data")


def say(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", file=sys.stderr, flush=True)


def run(step: str, args: List[str]) -> None:
    env = dict(os.environ, NLTK_DATA=NLTK_DIR, PYTHONIOENCODING="utf-8")
    say(step, "python " + " ".join(args))
    t0 = time.time()
    proc = subprocess.run([sys.executable] + args, cwd=HERE, env=env)
    if proc.returncode != 0:
        say(step, f"failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)
    say(step, f"done in {time.time() - t0:.0f}s")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_postings_sha(index_dir: str, canonical: bool) -> None:
    sha = sha256_file(os.path.join(index_dir, "postings.txt"))
    if canonical:
        info = {"postings_sha256": sha, "platform": platform.platform(), "python": platform.python_version(),
                "built_at": datetime.datetime.now().isoformat(timespec="seconds")}
        with open(CANONICAL_PATH, "w", encoding="utf-8", newline="\n") as f:
            json.dump(info, f, indent=2)
            f.write("\n")
        say("check", f"wrote {os.path.basename(CANONICAL_PATH)} (postings sha256 {sha[:16]}...)")
        return
    if not os.path.exists(CANONICAL_PATH):
        say("check", f"postings sha256 {sha[:16]}...; no canonical_index.json to compare with yet")
        return
    with open(CANONICAL_PATH, encoding="utf-8") as f:
        want = json.load(f)
    if want["postings_sha256"] == sha:
        say("check", "postings sha256 matches the canonical build")
    else:
        say("check", f"WARNING: postings sha256 {sha[:16]}... differs from the canonical build "
                     f"{want['postings_sha256'][:16]}... ({want.get('platform', '?')}); expected across platforms "
                     f"(lxml text extraction differs), results may differ slightly")


def check_token_ranks(index_dir: str, lock_path: str = LOCK_PATH) -> List[str]:
    """Problems found: each lock entry's docs must be the top 2 for its token, with ltc.ltc and with BM25."""
    sys.path.insert(0, HERE)
    import nltk
    nltk.data.path.insert(0, NLTK_DIR)
    from index_reader import IndexReader
    from query_parser import parse_query
    from ranker import Ranker
    from weighting import Scheme

    with open(lock_path, encoding="utf-8") as f:
        lock = json.load(f)
    problems = []
    with IndexReader(index_dir) as idx:
        rankers = {"ltc.ltc": Ranker(idx), "bm25": Ranker(idx, scheme=Scheme(idx, "bm25"))}
        for e in lock["entries"]:
            q = parse_query(e["token"], idx.cfg)
            want = set(e["doc_ids"])
            for name, r in rankers.items():
                top = [d for d, _ in r.rank(q, top_k=len(want)).hits]
                ok = set(top) == want
                say("check", f"{'OK  ' if ok else 'FAIL'} {name:7} {e['token']!r}: top {top}, lock {sorted(want)}")
                if not ok:
                    problems.append(f"{e['token']!r} ({name}): top {top} != {sorted(want)}")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--warc", default=os.path.join(PARENT, "07.warc.gz"), help="original WARC (default ../07.warc.gz)")
    ap.add_argument("--out-warc", default=os.path.join(PARENT, "07_team27.warc.gz"))
    ap.add_argument("--index-dir", default=os.path.join(HERE, "output"))
    ap.add_argument("--lock", default=LOCK_PATH, help="target doc ids per token (default special_tokens.lock.json)")
    ap.add_argument("--tokens", default=os.path.join(HERE, "special_tokens.json"))
    ap.add_argument("--from", dest="start", choices=STEPS, default="inject", help="first step to run")
    ap.add_argument("--canonical", action="store_true", help="write canonical_index.json (the Windows build)")
    a = ap.parse_args(argv)
    todo = STEPS[STEPS.index(a.start):]
    t0 = time.time()

    if "inject" in todo:
        if not os.path.exists(a.warc):
            say("inject", f"{a.warc} not found; put 07.warc.gz next to {os.path.basename(HERE)}/ or use --warc")
            return 2
        args = ["inject_tokens.py", "-i", a.warc, "-o", a.out_warc, "--lock", a.lock, "--tokens", a.tokens]
        # first run (no lock yet): also refuse tokens already in an existing (original) index
        if not os.path.exists(a.lock) and os.path.exists(os.path.join(a.index_dir, "meta.json")):
            args += ["--check-index", a.index_dir]
        run("inject", args)
    if "index" in todo:
        run("index", ["hw1_indexer.py", "-i", a.out_warc, "-o", a.index_dir, "--no-progress"])
    if "prep" in todo:
        run("prep", ["prep.py", "-i", a.index_dir, "--warc", a.out_warc])
    if "extra" in todo:
        run("extra", ["prep_extra.py", "-i", a.index_dir, "--force"])
    if "doc_text" in todo:
        run("doc_text", ["-m", "rag.doc_store", "build", "--warc", a.out_warc, "--index-dir", a.index_dir])
    if "check" in todo:
        check_postings_sha(a.index_dir, a.canonical)
        problems = check_token_ranks(a.index_dir, a.lock)
        if problems:
            for p in problems:
                say("check", "FAIL " + p)
            return 1
    say("all", f"finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
