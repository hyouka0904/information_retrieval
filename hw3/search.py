import argparse
import os
import re
import sys
import time
from typing import List, Optional, TextIO, Tuple

# index access layer
from index_reader import IndexReader
# query string -> AST, and its error type
from query_parser import QuerySyntaxError, parse_query
# TF-IDF + cosine ranking
from ranker import Ranker
# spelling correction on the raw query
from spell import SpellCorrector
# weighting schemes (SMART variants, BM25)
from weighting import DEFAULT_B, DEFAULT_K1, DEFAULT_SCHEME, HELP as SCHEME_HELP, Scheme, SchemeError

# text shown by --help (the full explanation is in README.md)
HELP_TEXT = """HW2 command-line search: prints "<doc#> <TAB> <similarity score>" lines (Slide 5).

  python search.py                              interactive prompt "cmd> " (exit / quit / Ctrl-D / Ctrl-Z ends it)
  python search.py Hong Kong business           one query from the command line
  python search.py -f queries.txt -o run.txt    batch: one query per line ("qid:query" or just the query)"""
# default index folder: "output" next to this file, so the program works from any working directory
_DEFAULT_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# "51:obama family tree" -> qid "51", query "obama family tree" (TREC topic file style)
_QID_RE = re.compile(r"^\s*(\w+)\s*:\s*(.*)$")


# command-line options (every option is listed in README.md)
def build_arg_parser() -> argparse.ArgumentParser:
    # parser with the module docstring as help text
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # optional query words; empty means interactive or batch mode
    ap.add_argument("query", nargs="*", help="query to run once (omit for the interactive prompt)")
    # folder with the HW1 index and the prep files
    ap.add_argument("-i", "--index", default=_DEFAULT_INDEX, help="index folder (default: output/ next to search.py)")
    # batch input file
    ap.add_argument("-f", "--file", help="batch mode: file with one query per line ('qid:query' allowed)")
    # where results go
    ap.add_argument("-o", "--output", help="write results to this file instead of the screen")
    # result list length
    ap.add_argument("-k", "--top-k", type=int, default=10, help="results per query, 0 = all (default: 10)")
    # decimals of the printed score
    ap.add_argument("--digits", type=int, default=3, help="decimals of the score (default: 3, as on Slide 5)")
    # extra column with the ClueWeb09 id
    ap.add_argument("--trec-id", action="store_true", help="add the ClueWeb09 TREC-ID as a third column")
    # TREC run format for trec_eval
    ap.add_argument("--trec-run", metavar="RUN_NAME",
                    help="write TREC run lines 'qid Q0 trec_id rank score RUN_NAME' (for trec_eval)")
    # extra diagnostics
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print the parsed query, candidate count, dropped terms and time")
    # inexact top-k: index elimination by idf
    ap.add_argument("--idf-threshold", type=float, default=None,
                    help="index elimination: ignore query terms with idf below this value")
    # inexact top-k: minimum number of query terms per document
    ap.add_argument("--min-match", type=int, default=1,
                    help="index elimination (free text): documents must contain at least this many query terms")
    # inexact top-k: champion lists
    ap.add_argument("--champions", action="store_true", help="score only documents in the champion lists")
    # weighting scheme
    ap.add_argument("--scheme", default=DEFAULT_SCHEME, metavar="DDD.QQQ",
                    help="term weighting in SMART notation (document.query) or 'bm25'; default ltc.ltc = Slide 4. "
                         "tf: n l a b L, idf: n t p, normalization: n c (e.g. lnc.ltc, anc.atc, bm25)")
    # BM25 parameters: tf saturation
    ap.add_argument("--k1", type=float, default=DEFAULT_K1, help="BM25 k1 (default: 1.2)")
    # BM25 length normalization
    ap.add_argument("--b", type=float, default=DEFAULT_B, help="BM25 b (default: 0.75)")
    # inexact top-k: tiered index
    ap.add_argument("--tiered", action="store_true",
                    help="tiered index: score documents from the high-tf tiers, lower tiers only if fewer than k found")
    # inexact top-k: impact ordering
    ap.add_argument("--impact", action="store_true",
                    help="impact ordering: tf-ordered postings with max-score early termination")
    # inexact top-k: cluster pruning
    ap.add_argument("--cluster", metavar="B1,B2", default=None,
                    help="cluster pruning: score only followers of the B2 leaders nearest to the query, every "
                         "document following its B1 nearest leaders (B1 <= 3), e.g. --cluster 1,4")
    # exact scoring from the binary tf index instead of postings.txt
    ap.add_argument("--tf-index", action="store_true",
                    help="exact ranking, but read (doc, tf) from the binary tf index instead of parsing postings.txt")
    # spelling correction mode
    ap.add_argument("--spell", choices=("off", "suggest", "correct"), default="suggest",
                    help="misspelled query words: off, suggest a correction (default, results unchanged), "
                         "or correct them before searching")
    # Soundex candidates for spelling correction
    ap.add_argument("--phonetic", action="store_true",
                    help="spelling correction also considers words that sound alike (Soundex)")
    # cap on wildcard expansions used for scoring
    ap.add_argument("--wildcard-limit", type=int, default=30,
                    help="most frequent expansions of a wildcard used for scoring (default: 30)")
    # finished parser
    return ap


# print every setting once at start-up
def print_settings(args: argparse.Namespace, idx: IndexReader, out: TextIO) -> None:
    # every switch on one line each, so a run can be reproduced (same habit as HW1's CLI)
    lines = [
        f"index          : {args.index}  (N={idx.N:,}, vocab={idx.vocab_size:,}, log base {idx.log_base})",
        f"top-k          : {args.top_k if args.top_k > 0 else 'all'}",
        f"idf threshold  : {args.idf_threshold if args.idf_threshold is not None else 'off'}",
        f"min match      : {args.min_match if args.min_match > 1 else 'off'}",
        f"champion lists : {'on' if args.champions else 'off'}",
        f"weighting      : {args.scheme}" + (f" (k1={args.k1}, b={args.b})" if args.scheme == "bm25" else ""),
        f"tiered index   : {'on' if args.tiered else 'off'}",
        f"impact ordering: {'on' if args.impact else 'off'}",
        f"cluster pruning: {'b1=%d, b2=%d' % args.cluster if args.cluster else 'off'}",
        f"postings source: {'tf_index.bin' if (args.tf_index or args.tiered or args.impact or args.cluster) else 'postings.txt'}",
        f"wildcard limit : {args.wildcard_limit}",
        f"spelling       : {args.spell}{' + phonetic' if args.phonetic and args.spell != 'off' else ''}",
    ]
    # settings go to stderr so they never mix with result files
    for line in lines:
        # one setting
        print(line, file=out)


# run one query: spelling, parse, rank, format; returns (result lines, diagnostic lines)
def run_one(ranker: Ranker, text: str, args: argparse.Namespace, qid: str,
            speller: Optional[SpellCorrector] = None) -> Tuple[List[str], List[str]]:
    # result lines (doc / score) for the output
    out: List[str] = []
    # diagnostics (warnings, errors, stats), always shown on screen
    diag: List[str] = []
    # spelling correction before parsing
    if speller is not None:
        # corrected query and the list of changes
        fixed, changes = speller.correct_query(text)
        # surface vocabulary missing: say so once per query
        if speller.unavailable:
            # spelling correction cannot run
            diag.append("# spelling correction unavailable: output/surface_vocab.tsv not found")
        # something was corrected
        elif changes:
            # readable list of the changes
            pairs = ", ".join(f"{a} -> {b}" for a, b in changes)
            # correct mode: search for the corrected query
            if args.spell == "correct":
                # report what was changed
                diag.append(f"# spelling corrected: {pairs}")
                # the corrected query replaces the typed one
                text = fixed
            # suggest mode: results stay those of the typed query
            else:
                # suggestion only
                diag.append(f"# did you mean: {fixed}")
    # index used for TREC-IDs
    idx = ranker.idx
    # parse with the index's linguistic settings
    try:
        # AST + mode
        parsed = parse_query(text, idx.cfg)
    # malformed query
    except QuerySyntaxError as e:
        # report and return no results
        diag.append(f"# error: {e}")
        # nothing ranked
        return out, diag
    # parser notes (stopwords dropped, nothing to rank, ...)
    diag += [f"# warning: {w}" for w in parsed.warnings]
    # verbose: show how the query was understood
    if args.verbose:
        # indented parse result
        diag += ["# " + line for line in str(parsed).splitlines()]
    # start timer
    t0 = time.time()
    # rank with the chosen options
    try:
        # ranked hits
        res = ranker.rank(parsed, top_k=args.top_k, idf_threshold=args.idf_threshold,
                          min_match=args.min_match, champions=args.champions, tiered=args.tiered,
                          impact=args.impact, cluster=args.cluster, tf_index=args.tf_index)
    # champion lists requested but not built
    except FileNotFoundError as e:
        # report and return no results
        diag.append(f"# error: {e}")
        # nothing ranked
        return out, diag
    # elapsed milliseconds
    ms = (time.time() - t0) * 1000
    # one line per hit
    for rank, (d, score) in enumerate(res.hits, start=1):
        # TREC run format for trec_eval
        if args.trec_run:
            # qid Q0 docno rank score tag
            out.append(f"{qid} Q0 {idx.doc_label(d)} {rank} {score:.6f} {args.trec_run}")
        # Slide 5 format, optionally with the TREC-ID
        else:
            # "<doc#> <TAB> <score>"
            line = f"{d}\t{score:.{args.digits}f}"
            # extra column on request
            if args.trec_id:
                # append the ClueWeb09 id
                line += f"\t{idx.doc_label(d)}"
            # finished line
            out.append(line)
    # an empty result is stated explicitly
    if not res.hits:
        # diagnostic only, so result files stay parseable
        diag.append("# no results")
    # verbose: candidate count, used / dropped terms, matcher counters, time
    if args.verbose:
        # summary line
        diag.append(f"# {res.num_candidates:,} candidates, terms {res.terms_used}, "
                    f"dropped {res.terms_dropped or '-'}, {ranker.matcher.stats}, "
                    f"{res.postings_scored:,} postings scored, {ms:.1f} ms")
        # remarks of the weighting / inexact top-k code
        diag += [f"# {note}" for note in res.notes]
        # champion lists were requested but too small for this query
        if res.champions_fallback:
            # say that the exact candidate set was used
            diag.append(f"# the pruned candidate set gave fewer than {args.top_k} documents; search was widened")
    # results and diagnostics
    return out, diag


# yield (qid, query) from a batch file; blank lines and lines starting with '#' are skipped
def iter_batch(path: str):
    # LF / CRLF both fine for a hand-written query file
    with open(path, encoding="utf-8") as f:
        # line numbers become qids when a line has no "qid:" prefix
        for n, line in enumerate(f, start=1):
            # strip newline and surrounding spaces
            line = line.strip()
            # ignore empty lines and comments
            if not line or line.startswith("#"):
                # next line
                continue
            # "qid:query" form
            m = _QID_RE.match(line)
            # an explicit qid that is followed by a non-empty query
            if m and m.group(2):
                # given qid and query
                yield m.group(1), m.group(2)
            # plain query line
            else:
                # line number as qid
                yield str(n), line


# program entry: load the index, then batch, one-shot or interactive mode; returns the exit code
def main(argv: Optional[List[str]] = None) -> int:
    # parse command-line options
    args = build_arg_parser().parse_args(argv)
    # never crash on characters the console cannot show (e.g. CJK on a cp950 terminal)
    sys.stdout.reconfigure(errors="replace")
    # same for diagnostics
    sys.stderr.reconfigure(errors="replace")
    # "--cluster 1,4" -> (1, 4); a single number means b1 = 1
    if args.cluster:
        # a malformed value raises ValueError or fails the range check
        try:
            # the comma-separated numbers
            nums = [int(x) for x in args.cluster.split(",")]
            # one number is b2 with b1 = 1, two numbers are b1, b2
            args.cluster = (1, nums[0]) if len(nums) == 1 else (nums[0], nums[1])
            # at most two numbers, 1 <= b1 <= 3 (followers store 3 leaders), b2 >= 1
            assert len(nums) <= 2 and 1 <= args.cluster[0] <= 3 and args.cluster[1] >= 1
        # not two valid numbers
        except (ValueError, AssertionError):
            # show the expected form
            print("--cluster expects B1,B2 with 1 <= B1 <= 3 and B2 >= 1, e.g. --cluster 1,4", file=sys.stderr)
            # usage error exit code
            return 2
    # load the index (meta, offsets, norms, TREC-IDs)
    try:
        # read-only index access
        idx = IndexReader(args.index)
    # missing files or a stale offsets table
    except (FileNotFoundError, RuntimeError) as e:
        # explain and stop
        print(f"cannot open index: {e}", file=sys.stderr)
        # non-zero exit code
        return 1
    # weighting scheme (ltc.ltc unless --scheme says otherwise)
    try:
        # parse the scheme name
        scheme = Scheme(idx, args.scheme, args.k1, args.b)
    # unknown letters or shape
    except SchemeError as e:
        # the problem and the notation help
        print(str(e), SCHEME_HELP, sep="\n", file=sys.stderr)
        # release the index before leaving
        idx.close()
        # usage error exit code
        return 2
    # ranker with a cap on wildcard expansions
    ranker = Ranker(idx, wildcard_limit=args.wildcard_limit, scheme=scheme)
    # spelling corrector sharing the matcher's surface vocabulary (loaded only when a word needs it)
    speller = None if args.spell == "off" else SpellCorrector(idx, ranker.matcher.surface_vocab, phonetic=args.phonetic)
    # result destination: a file or the screen
    out = open(args.output, "w", encoding="utf-8", newline="\n") if args.output else sys.stdout
    # always close files and the index
    try:
        # show the settings once at start-up
        print_settings(args, idx, sys.stderr)

        # batch mode
        if args.file:
            # every query in the file
            for qid, text in iter_batch(args.file):
                # run it
                lines, diag = run_one(ranker, text, args, qid, speller)
                # plain format: echo the query the way the slide shows it
                if not args.trec_run:
                    # "cmd> query" header
                    print(f"cmd> {text}", file=out)
                # result lines
                for line in lines:
                    # one hit
                    print(line, file=out)
                # blank separator between queries in plain format
                if not args.trec_run:
                    # empty line
                    print(file=out)
                # diagnostics on screen, prefixed with the qid
                for line in diag:
                    # one diagnostic
                    print(f"[{qid}] {line}", file=sys.stderr)
            # done
            return 0

        # one-shot mode: the query words joined back together
        if args.query:
            # the full query string
            text = " ".join(args.query)
            # run it as qid 1
            lines, diag = run_one(ranker, text, args, "1", speller)
            # diagnostics first, on the screen
            for line in diag:
                # one diagnostic
                print(line, file=sys.stderr)
            # result lines
            for line in lines:
                # one hit
                print(line, file=out)
            # done
            return 0

        # interactive mode; qids count the queries typed
        qid = 0
        # read until EOF or an exit command
        while True:
            # prompt like the slide; EOF (Ctrl-D / Ctrl-Z) ends the session
            try:
                # one line from the user
                text = input("cmd> ").strip()
            # end of input
            except (EOFError, KeyboardInterrupt):
                # finish the prompt line
                print(file=sys.stderr)
                # leave the loop
                break
            # empty line: prompt again
            if not text:
                # next prompt
                continue
            # explicit exit commands
            if text.lower() in ("exit", "quit"):
                # leave the loop
                break
            # next query number
            qid += 1
            # run it
            lines, diag = run_one(ranker, text, args, str(qid), speller)
            # diagnostics on screen
            for line in diag:
                # one diagnostic
                print(line, file=sys.stderr)
            # result lines
            for line in lines:
                # one hit
                print(line, file=out)
            # keep a result file up to date after every query
            out.flush()
        # done
        return 0
    # cleanup
    finally:
        # close the result file (never close stdout)
        if out is not sys.stdout:
            # flush and close
            out.close()
        # close the postings handle
        idx.close()


# run as a script
if __name__ == "__main__":
    # exit code of main()
    sys.exit(main())
