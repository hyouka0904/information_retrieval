"""Retrieval evaluation on the graded prompt set: hit@1, hit@k and time per retriever, by prompt type and language.

Every retriever the registry can build is evaluated (lexical now; vector / hybrid once F1 / F2 exist), so the tables
grow by themselves. A prompt is a hit at k when one of its expected_docs is among the first k passages.

    python evaluate_retrieval.py                       # prompts/eval_prompts.jsonl -> results/retrieval.md
    python evaluate_retrieval.py --retriever lexical -k 10 -o results/retrieval_k10.md
"""
import argparse
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rag.config import load_config  # noqa: E402
from rag.registry import RETRIEVERS, available_retrievers, make_retriever  # noqa: E402

PROMPTS_PATH = os.path.join(HERE, "prompts", "eval_prompts.jsonl")
OUT_PATH = os.path.join(HERE, "results", "retrieval.md")
REQUIRED = ("id", "type", "lang", "prompt", "expected_docs", "expected_answer")


def load_prompts(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            p = json.loads(line)
            missing = [k for k in REQUIRED if k not in p]
            if missing:
                raise ValueError(f"{path}:{n}: missing {', '.join(missing)}")
            out.append(p)
    return out


def _rate(rows: List[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _summary(rows: List[dict]) -> dict:
    return {"n": len(rows), "hit@1": _rate(rows, "hit1"), "hit@k": _rate(rows, "hitk"),
            "mean_ms": sum(r["ms"] for r in rows) / len(rows) if rows else 0.0}


def evaluate(prompts: List[dict], retrievers: Dict[str, object], k: int = 5) -> List[dict]:
    """Per retriever: {"retriever", "k", "n", "hit@1", "hit@k", "hit@5" (when k == 5), "mean_ms",
    "by_type": {type: summary}, "by_lang": {lang: summary}, "per_prompt": [{id, type, lang, rank, top, ms, ...}]}.
    rank is the 1-based position of the first expected doc in the top k, or None."""
    results = []
    for name, r in retrievers.items():
        rows = []
        for p in prompts:
            t0 = time.perf_counter()
            hits = r.search(p["prompt"], k)
            ms = (time.perf_counter() - t0) * 1000
            top = [h.doc_id for h in hits]
            want = set(p["expected_docs"])
            rank = next((i for i, d in enumerate(top, 1) if d in want), None)
            rows.append({"id": p["id"], "type": p["type"], "lang": p["lang"], "prompt": p["prompt"],
                         "expected_docs": p["expected_docs"], "top": top, "rank": rank,
                         "hit1": rank == 1, "hitk": rank is not None, "ms": ms})
        res = {"retriever": name, "k": k, **_summary(rows),
               "by_type": {t: _summary([x for x in rows if x["type"] == t]) for t in sorted({x["type"] for x in rows})},
               "by_lang": {g: _summary([x for x in rows if x["lang"] == g]) for g in sorted({x["lang"] for x in rows})},
               "per_prompt": rows}
        if k == 5:
            res["hit@5"] = res["hit@k"]
        results.append(res)
    return results


def build_retrievers(cfg: dict, names: Optional[List[str]] = None) -> Tuple[Dict[str, object], List[str]]:
    """(name -> retriever for every buildable name, names listed as not built)."""
    avail = available_retrievers(cfg)
    wanted = names or list(RETRIEVERS)
    built, not_built = {}, []
    for n in wanted:
        if n in avail:
            built[n] = make_retriever(n, cfg)
        else:
            not_built.append(n)
    return built, not_built


def _frac(x: float, n: int) -> str:
    return f"{round(x * n)}/{n} ({x:.0%})"


def to_markdown(results: List[dict], not_built: List[str], prompts_path: str, cfg: dict) -> str:
    k = results[0]["k"] if results else 5
    shown = os.path.relpath(prompts_path, HERE) if os.path.abspath(prompts_path).startswith(HERE) else prompts_path
    lines = ["# Retrieval evaluation", "",
             f"- prompts: `{shown}`; date {datetime.date.today().isoformat()}",
             f"- hit@k: an expected document is among the first k passages (k = {k})",
             f"- lexical settings: `{json.dumps(cfg.get('lexical', {}))}`"]
    if not_built:
        lines.append(f"- not built: {', '.join(not_built)}")
    lines += ["", "## Summary", "", f"| retriever | group | n | hit@1 | hit@{k} | mean ms |", "|---|---|---|---|---|---|"]
    for r in results:
        groups = [("all", r)] + [(f"type={t}", s) for t, s in r["by_type"].items()] + \
                 [(f"lang={g}", s) for g, s in r["by_lang"].items()]
        for label, s in groups:
            lines.append(f"| {r['retriever']} | {label} | {s['n']} | {_frac(s['hit@1'], s['n'])} | "
                         f"{_frac(s['hit@k'], s['n'])} | {s['mean_ms']:.1f} |")
    for r in results:
        lines += ["", f"## Per prompt: {r['retriever']}", "",
                  f"| id | type | lang | prompt | expected | rank | top {k} |", "|---|---|---|---|---|---|---|"]
        for x in r["per_prompt"]:
            prompt = x["prompt"].replace("|", "\\|")
            lines.append(f"| {x['id']} | {x['type']} | {x['lang']} | {prompt} | "
                         f"{', '.join(map(str, x['expected_docs']))} | {x['rank'] or '-'} | "
                         f"{', '.join(map(str, x['top'])) or '(none)'} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-p", "--prompts", default=PROMPTS_PATH)
    ap.add_argument("-o", "--output", default=OUT_PATH)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--retriever", action="append", choices=list(RETRIEVERS),
                    help="evaluate only these (repeatable; default: every buildable retriever)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    prompts = load_prompts(a.prompts)
    built, not_built = build_retrievers(cfg, a.retriever)
    if not built:
        print("no retriever can be built: " + ", ".join(not_built), file=sys.stderr)
        return 2
    results = evaluate(prompts, built, a.k)
    md = to_markdown(results, not_built, a.prompts, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    for r in results:
        print(f"{r['retriever']:8} n={r['n']} hit@1={r['hit@1']:.2f} hit@{a.k}={r['hit@k']:.2f} "
              f"mean {r['mean_ms']:.1f} ms", file=sys.stderr)
    print(f"wrote {a.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
