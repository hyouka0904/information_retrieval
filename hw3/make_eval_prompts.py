"""Assemble prompts/eval_prompts.jsonl: 6 special-token prompts (from special_tokens.lock.json) + 12 hand-written
collection prompts (prompts/collection_prompts.jsonl), numbered P01..P18.

collection_prompts.jsonl has one JSON object per line, written while reading the pages with
`python -m rag.doc_store show <doc_id>`:

    {"lang": "en", "prompt": "...", "expected_docs": [4821], "expected_answer": ["1791"], "reworded": false}

Rules (checked): 12 prompts, exactly 2 with "lang": "zh", exactly 2 with "reworded": true (worded differently from
the page), every expected_docs id within 1..N of the index when output/ exists.

    python make_eval_prompts.py --init     # write a 12-line template to fill in (never overwrites)
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


TODO = "TODO"


def template() -> List[dict]:
    """12 placeholder rows with the required mix: 8 English, 2 English reworded, 2 Chinese."""
    kinds = [("en", False)] * 8 + [("en", True)] * 2 + [("zh", False)] * 2
    return [{"lang": lang, "prompt": TODO, "expected_docs": [], "expected_answer": [], "reworded": rw}
            for lang, rw in kinds]


def write_template(path: str) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; edit it instead")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in template():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_collection(path: str) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path} line {i} is not valid JSON: {e.msg}") from None
    return items


def check_collection(items: List[dict], n_docs: Optional[int]) -> None:
    problems = []
    if len(items) != 12:
        problems.append(f"{len(items)} collection prompts, expected 12")
    if sum(p.get("lang") == "zh" for p in items) != 2:
        problems.append('exactly 2 prompts need "lang": "zh"')
    if sum(bool(p.get("reworded")) for p in items) != 2:
        problems.append('exactly 2 prompts need "reworded": true')
    for i, p in enumerate(items, 1):
        if p.get("prompt") == TODO:
            problems.append(f"line {i}: still the template; fill in prompt, expected_docs and expected_answer")
            continue
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
    ap.add_argument("--init", action="store_true", help="write a 12-line template to --collection and stop")
    a = ap.parse_args(argv)
    if a.init:
        try:
            write_template(a.collection)
        except FileExistsError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"wrote a 12-line template to {a.collection}; replace every TODO row with a real prompt "
              f"(A_LINE_TASKS.md step 6)", file=sys.stderr)
        return 0
    if not os.path.exists(a.lock):
        print(f"error: {a.lock} is missing; run 'python rebuild_index.py' first", file=sys.stderr)
        return 1
    if not os.path.exists(a.collection):
        print(f"error: {a.collection} is missing; run 'python make_eval_prompts.py --init' and fill in the "
              f"12 rows (A_LINE_TASKS.md step 6)", file=sys.stderr)
        return 1
    with open(a.lock, encoding="utf-8") as f:
        lock = json.load(f)
    try:
        collection = read_collection(a.collection)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
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
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {a.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
