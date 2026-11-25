"""Assemble prompts/eval_prompts.jsonl: 6 special-token prompts (from special_tokens.lock.json) + 12 hand-written
collection prompts (prompts/collection_prompts.jsonl), numbered P01..P18.

collection_prompts.jsonl has one JSON object per line, written while reading the pages with
`python -m rag.doc_store show <doc_id>`:

    {"lang": "en", "prompt": "...", "expected_docs": [4821], "expected_answer": ["1791"], "reworded": false}

Rules (checked): 12 prompts, exactly 2 with "lang": "zh", exactly 2 with "reworded": true (worded differently from
the page), every expected_docs id within 1..N of the index when output/ exists.

    python make_eval_prompts.py
"""
import argparse
import json
import os
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(HERE, "special_tokens.lock.json")
COLLECTION_PATH = os.path.join(HERE, "prompts", "collection_prompts.jsonl")
OUT_PATH = os.path.join(HERE, "prompts", "eval_prompts.jsonl")
INDEX_DIR = os.path.join(HERE, "output")

# token -> (question form, expected answer strings); every fact is invented, so no LLM knows it without RAG
QUESTIONS = {
    "115598095": ("What is the project codename for 115598095?", ["BLUE-HERON-7"]),
    "115598063": ("What is the project codename for 115598063?", ["AMBER-LYNX-3"]),
    "zorvian lantern": ("When is the Zorvian Lantern Festival held?", ["March 14"]),
}


def special_prompts(lock: dict) -> List[dict]:
    out = []
    for e in lock["entries"]:
        if e["token"] not in QUESTIONS:
            raise ValueError(f"no question form for token {e['token']!r}; add it to QUESTIONS")
        question, answer = QUESTIONS[e["token"]]
        for prompt in (e["token"], question):
            out.append({"type": "special", "lang": "en", "prompt": prompt, "expected_docs": list(e["doc_ids"]),
                        "expected_answer": answer})
    return out


def check_collection(items: List[dict], n_docs: Optional[int]) -> None:
    problems = []
    if len(items) != 12:
        problems.append(f"{len(items)} collection prompts, expected 12")
    if sum(p.get("lang") == "zh" for p in items) != 2:
        problems.append('exactly 2 prompts need "lang": "zh"')
    if sum(bool(p.get("reworded")) for p in items) != 2:
        problems.append('exactly 2 prompts need "reworded": true')
    for i, p in enumerate(items, 1):
        for key in ("lang", "prompt", "expected_docs", "expected_answer"):
            if not p.get(key):
                problems.append(f"line {i}: missing {key}")
        if p.get("lang") not in ("en", "zh"):
            problems.append(f"line {i}: lang must be en or zh")
        if n_docs is not None:
            bad = [d for d in p.get("expected_docs", []) if not 1 <= d <= n_docs]
            if bad:
                problems.append(f"line {i}: doc ids {bad} outside 1..{n_docs}")
    if problems:
        raise ValueError("; ".join(problems))


def assemble(lock: dict, collection: List[dict], n_docs: Optional[int] = None) -> List[dict]:
    check_collection(collection, n_docs)
    rows = special_prompts(lock) + [
        {"type": "collection", "lang": p["lang"], "prompt": p["prompt"], "expected_docs": p["expected_docs"],
         "expected_answer": p["expected_answer"], **({"reworded": True} if p.get("reworded") else {})}
        for p in collection]
    return [{"id": f"P{i:02d}", **r} for i, r in enumerate(rows, 1)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock", default=LOCK_PATH)
    ap.add_argument("--collection", default=COLLECTION_PATH)
    ap.add_argument("-o", "--output", default=OUT_PATH)
    a = ap.parse_args(argv)
    with open(a.lock, encoding="utf-8") as f:
        lock = json.load(f)
    with open(a.collection, encoding="utf-8") as f:
        collection = [json.loads(line) for line in f if line.strip()]
    n_docs = None
    meta = os.path.join(INDEX_DIR, "meta.json")
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            n_docs = json.load(f)["N"]
    try:
        rows = assemble(lock, collection, n_docs)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with open(a.output, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {a.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
