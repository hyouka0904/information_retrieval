import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

# folder this script lives in; every path below is relative to it
ROOT = os.path.dirname(os.path.abspath(__file__))
# index and preprocessing output, rebuilt from the WARC (never committed)
OUTPUT_DIR = os.path.join(ROOT, "output")
# ranked lists that go into the submission
RESULTS_DIR = os.path.join(ROOT, "results")
# packaged submissions
DIST_DIR = os.path.join(ROOT, "dist")
# bundled NLTK stopwords, so the HW1 indexer runs offline
NLTK_DIR = os.path.join(ROOT, "nltk_data")
# default location of the ClueWeb09 WARC file
DEFAULT_WARC = os.path.join(ROOT, "07.warc.gz")
# upload limit of the course site
UPLOAD_LIMIT = 64 * 1000 * 1000

# result file name -> search.py options (every run writes results/<name>)
RESULT_RUNS = [
    ("results.txt", ["-f", "queries.txt"]),
    ("results_champions.txt", ["-f", "queries.txt", "--champions"]),
    ("results_spell_correct.txt", ["-f", "queries.txt", "--spell", "correct"]),
    ("results_tiered.txt", ["-f", "queries.txt", "--tiered"]),
    ("results_impact.txt", ["-f", "queries.txt", "--impact"]),
    ("results_cluster_3,8.txt", ["-f", "queries.txt", "--cluster", "3,8"]),
    ("results_scheme_lnc.ltc.txt", ["-f", "queries.txt", "--scheme", "lnc.ltc"]),
    ("results_scheme_anc.atc.txt", ["-f", "queries.txt", "--scheme", "anc.atc"]),
    ("results_scheme_bm25.txt", ["-f", "queries.txt", "--scheme", "bm25"]),
    ("results_bm25_impact.txt", ["-f", "queries.txt", "--scheme", "bm25", "--impact"]),
    ("webtrack_ltc.txt", ["-f", "queries_webtrack.txt", "--trec-id"]),
    ("webtrack_bm25.txt", ["-f", "queries_webtrack.txt", "--scheme", "bm25", "--trec-id"]),
]

# top-level files copied into the submission zip
PACKAGE_FILES = [
    "README.md", "README.pdf", "requirements.txt", "queries.txt", "queries_webtrack.txt", "build.py",
    "hw1_indexer.py", "prep.py", "prep_extra.py", "pack_index.py", "hw2_index_prep.ipynb", "index_reader.py",
    "query_parser.py", "matcher.py", "ranker.py", "weighting.py", "tf_index.py", "clusters.py", "kgram.py",
    "spell.py", "search.py", "ui.py", "evaluate.py",
]
# folders copied into the submission zip as a whole
PACKAGE_DIRS = ["nltk_data", "results", "output"]
# output/ files left out of the zip: postings.txt ships packed, the others are rebuilt automatically on first use
PACKAGE_EXCLUDE = {"postings.txt", "tf_index.bin", "tf_index.dir", "tf_index.json", "kgram.bin", "kgram.json",
                   "doc_text.tsv"}
# output/ files of prep.py
PREP_OUTPUT = ["offsets.tsv", "doc_norms.tsv", "doc_lengths.tsv", "doc_ids.tsv", "champions.tsv",
               "champions_bm25.tsv", "surface_vocab.tsv", "meta.json"]
# output/ files of prep_extra.py that go into the zip
EXTRA_OUTPUT = ["doc_stats.tsv", "cluster_leaders.tsv", "cluster_followers.tsv", "clusters.json"]
# output/ files of pack_index.py
PACK_OUTPUT = ["postings.delta.xz", "postings.pack.json"]
# every output/ file the zip must contain
REQUIRED_OUTPUT = ["dictionary.txt"] + PREP_OUTPUT + EXTRA_OUTPUT + PACK_OUTPUT

# steps run by "all", in order ("evaluate" is separate: it takes several minutes and only refreshes the report)
ALL_STEPS = ["index", "prep", "extra", "pack", "results", "package"]

# text shown by --help
HELP_TEXT = """HW2 build script: rebuild everything from the WARC and package the submission.
steps:
  index      hw1_indexer.py: 07.warc.gz -> output/dictionary.txt, output/postings.txt
  prep       prep.py: offsets, norms, lengths, TREC-IDs, champion lists, surface vocabulary, meta.json
  extra      prep_extra.py: tf index, doc_stats.tsv, cluster pruning, k-gram index
  pack       pack_index.py pack: postings.txt -> postings.delta.xz (the zip must stay under 64 MB)
  results    every file of results/ from queries.txt and queries_webtrack.txt
  evaluate   evaluate.py all: results/evaluation.md (correctness and speed; several minutes, not part of "all")
  package    dist/<ID>_HW2.zip
  clean      delete output/ and dist/
examples:
  python build.py all --id 115598095      every step; steps whose output is already up to date are skipped
  python build.py clean all               start from nothing (use this after switching to another index)
  python build.py results evaluate        refresh results/ and the evaluation report only"""


# print one progress line with the step name
def say(step: str, msg: str) -> None:
    # flush so the line shows before a long subprocess starts
    print(f"[{step}] {msg}", flush=True)


# run a Python script of this project with the same interpreter; stop the build if it fails
def run_python(step: str, args: list, env_extra: dict = None) -> None:
    # child environment: ours plus any extra variables
    env = dict(os.environ, **(env_extra or {}))
    # show the exact command so it can be rerun by hand
    say(step, "$ python " + " ".join(args))
    # start timer
    t0 = time.time()
    # run in the project folder so relative paths in the scripts work
    proc = subprocess.run([sys.executable] + args, cwd=ROOT, env=env)
    # a failed step makes every later step meaningless
    if proc.returncode != 0:
        # report and stop with the child's exit code
        say(step, f"FAILED (exit code {proc.returncode})")
        # leave the build
        sys.exit(proc.returncode)
    # duration of the step
    say(step, f"done in {time.time() - t0:.1f}s")


# output/<name>
def out(name: str) -> str:
    # absolute path inside output/
    return os.path.join(OUTPUT_DIR, name)


# True when every target exists and is at least as new as every existing source
def up_to_date(targets: list, sources: list) -> bool:
    # a missing target always needs building
    if not all(os.path.exists(t) for t in targets):
        # build it
        return False
    # oldest target decides
    t = min(os.path.getmtime(t) for t in targets)
    # every existing source must be older
    return all(os.path.getmtime(s) <= t for s in sources if os.path.exists(s))


# stop the build with a message
def fail(step: str, msg: str) -> None:
    # explain
    say(step, msg)
    # non-zero exit code
    sys.exit(1)


# step "index": HW1 indexer over the WARC -> output/dictionary.txt, output/postings.txt
def step_index(args: argparse.Namespace) -> None:
    # main output of the indexer
    postings = out("postings.txt")
    # the WARC is the only input
    if not os.path.exists(args.warc):
        # an existing index is still usable without the WARC
        if os.path.exists(postings) and not args.force:
            # keep it
            say("index", f"{args.warc} not found; keeping the existing index")
            # nothing to do
            return
        # cannot build at all
        fail("index", f"{args.warc} not found; put 07.warc.gz next to build.py or pass --warc PATH")
    # index newer than the WARC and the indexer: skip (make-style)
    if not args.force and up_to_date([postings, out("dictionary.txt")], [args.warc, os.path.join(ROOT, "hw1_indexer.py")]):
        # report the skip
        say("index", "output/postings.txt is up to date (use --force, or 'clean' first, to rebuild)")
        # nothing to do
        return
    # HW1's indexer loads NLTK stopwords at import time, so point NLTK at the bundled copy
    run_python("index", ["hw1_indexer.py", "-i", args.warc, "-o", OUTPUT_DIR, "--no-progress"],
               {"NLTK_DATA": NLTK_DIR})


# step "prep": prep.py over the index (with the WARC for N, TREC-IDs and the surface vocabulary)
def step_prep(args: argparse.Namespace) -> None:
    # prep.py needs the HW1 index
    if not os.path.exists(out("postings.txt")):
        # explain which step comes first
        fail("prep", "output/postings.txt missing; run 'python build.py index' first")
    # every output newer than the index and prep.py: skip
    if not args.force and up_to_date([out(f) for f in PREP_OUTPUT], [out("postings.txt"), os.path.join(ROOT, "prep.py")]):
        # report the skip
        say("prep", "prep.py output is up to date (use --force to rerun)")
        # nothing to do
        return
    # same index folder
    cmd = ["prep.py", "-i", OUTPUT_DIR]
    # with the WARC every table is rebuilt
    if os.path.exists(args.warc):
        # pass its location
        cmd += ["--warc", args.warc]
    # without it prep.py keeps doc_ids.tsv / surface_vocab.tsv only if they belong to this index
    else:
        # tell prep.py not to look for it
        cmd += ["--no-warc"]
    # run the preprocessing (it verifies its own output at the end)
    run_python("prep", cmd)


# step "extra": prep_extra.py (tf index, doc_stats.tsv, clusters, k-gram index)
def step_extra(args: argparse.Namespace) -> None:
    # prep_extra.py reads prep.py's output
    if not os.path.exists(out("meta.json")):
        # explain which step comes first
        fail("extra", "output/meta.json missing; run 'python build.py prep' first")
    # files this step must produce
    targets = [out(f) for f in EXTRA_OUTPUT + ["tf_index.json", "kgram.json"]]
    # everything newer than meta.json (written last by prep.py): skip
    if not args.force and up_to_date(targets, [out("meta.json"), os.path.join(ROOT, "prep_extra.py")]):
        # report the skip
        say("extra", "prep_extra.py output is up to date (use --force to rerun)")
        # nothing to do
        return
    # --force: prep_extra.py keeps an existing doc_stats.tsv otherwise, even when the index changed
    run_python("extra", ["prep_extra.py", "-i", OUTPUT_DIR, "--force"])


# SHA-256 of a file
def sha256_of(path: str) -> str:
    # running hash
    h = hashlib.sha256()
    # binary mode
    with open(path, "rb") as f:
        # 1 MiB chunks until EOF
        for chunk in iter(lambda: f.read(1 << 20), b""):
            # add the chunk
            h.update(chunk)
    # hex digest
    return h.hexdigest()


# True when postings.pack.json describes the postings.txt on disk (same size and SHA-256)
def pack_matches_postings() -> bool:
    # both files are needed for the comparison
    if not (os.path.exists(out("postings.pack.json")) and os.path.exists(out("postings.txt"))):
        # cannot confirm
        return False
    # read the manifest
    with open(out("postings.pack.json"), encoding="utf-8") as f:
        # expected size and hash
        info = json.load(f)
    # cheap size check first, then the hash
    return info["postings_bytes"] == os.path.getsize(out("postings.txt")) and info["sha256"] == sha256_of(out("postings.txt"))


# step "pack": postings.txt -> postings.delta.xz + postings.pack.json
def step_pack(args: argparse.Namespace) -> None:
    # the packed form is made from postings.txt
    if not os.path.exists(out("postings.txt")):
        # explain which step comes first
        fail("pack", "output/postings.txt missing; run 'python build.py index' first")
    # packed file present and describing exactly this postings.txt: skip
    if not args.force and os.path.exists(out("postings.delta.xz")) and pack_matches_postings():
        # report the skip
        say("pack", "postings.delta.xz matches postings.txt (use --force to repack)")
        # nothing to do
        return
    # pack and verify by restoring once (about 2.5 minutes)
    run_python("pack", ["pack_index.py", "pack", "-i", OUTPUT_DIR])


# step "results": every ranked list of results/
def step_results(args: argparse.Namespace) -> None:
    # the search needs a finished preprocessing
    if not os.path.exists(out("meta.json")):
        # explain which step comes first
        fail("results", "output/meta.json missing; run 'python build.py prep' first")
    # make sure the folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # one search.py run per result file; they take seconds, so they always rerun
    for name, opts in RESULT_RUNS:
        # batch mode with the run's options
        run_python("results", ["search.py"] + opts + ["-o", os.path.join("results", name)])


# step "evaluate": results/evaluation.md
def step_evaluate(args: argparse.Namespace) -> None:
    # the evaluation reads the full index and the extra files
    if not os.path.exists(out("meta.json")):
        # explain which step comes first
        fail("evaluate", "output/meta.json missing; run 'python build.py prep extra' first")
    # correctness and speed; writes results/evaluation.md
    run_python("evaluate", ["evaluate.py", "all", "-i", OUTPUT_DIR])


# step "package": dist/<id>_HW2.zip
def step_package(args: argparse.Namespace) -> None:
    # every output file must exist, or the TA cannot run search.py
    missing = [f for f in REQUIRED_OUTPUT if not os.path.exists(out(f))]
    # incomplete preprocessing
    if missing:
        # list what is missing
        fail("package", "output/ is incomplete (missing " + ", ".join(missing) + "); run 'python build.py all' first")
    # the packed postings must belong to the shipped offsets.tsv, or every seek on the TA's machine would be wrong
    if os.path.exists(out("postings.txt")) and not pack_matches_postings():
        # the classic mistake: index rebuilt, pack forgotten
        fail("package", "postings.delta.xz does not match postings.txt; run 'python build.py pack' first")
    # every result file must exist as well
    missing = [n for n, _ in RESULT_RUNS if not os.path.exists(os.path.join(RESULTS_DIR, n))]
    # incomplete results
    if missing:
        # list what is missing
        fail("package", "results/ is incomplete (missing " + ", ".join(missing) + "); run 'python build.py results' first")
    # file and folder name required by the assignment: <ID>_HW2
    base = f"{args.id}_HW2"
    # make sure dist/ exists
    os.makedirs(DIST_DIR, exist_ok=True)
    # final zip path
    zip_path = os.path.join(DIST_DIR, base + ".zip")
    # (path on disk, path inside the zip) for every file to pack
    entries = []
    # top-level files
    for name in PACKAGE_FILES:
        # every listed file must exist
        if not os.path.exists(os.path.join(ROOT, name)):
            # a missing source file is a broken checkout
            fail("package", f"{name} not found")
        # pack under <ID>_HW2/
        entries.append((os.path.join(ROOT, name), f"{base}/{name}"))
    # whole folders
    for folder in PACKAGE_DIRS:
        # walk the folder in a fixed order so two builds give the same zip listing
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, folder)):
            # skip caches and sort for a stable order
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", ".ipynb_checkpoints"))
            # every file in this folder
            for fn in sorted(filenames):
                # big or rebuildable output files and leftovers of interrupted runs stay out
                if folder == "output" and (fn in PACKAGE_EXCLUDE or fn.endswith((".tmp", ".check"))):
                    # skip it
                    continue
                # absolute path on disk
                src = os.path.join(dirpath, fn)
                # path inside the zip, always with forward slashes
                arc = base + "/" + os.path.relpath(src, ROOT).replace(os.sep, "/")
                # add to the list
                entries.append((src, arc))
    # write to a temporary name first, so a broken run never leaves a half-written zip with the real name
    tmp_path = zip_path + ".part"
    # report
    say("package", f"writing {os.path.relpath(zip_path, ROOT)} ({len(entries)} files)")
    # deflate-compressed archive
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # every entry
        for src, arc in entries:
            # add the file
            zf.write(src, arc)
    # publish the finished zip
    os.replace(tmp_path, zip_path)
    # final size
    size = os.path.getsize(zip_path)
    # report size against the upload limit
    say("package", f"done: {size / 1e6:.1f} MB" + (" (OVER the 64 MB upload limit!)" if size > UPLOAD_LIMIT else ""))


# step "clean": delete everything the build generates (results/ is kept: it is committed)
def step_clean(args: argparse.Namespace) -> None:
    # generated folders
    for folder in (OUTPUT_DIR, DIST_DIR):
        # only if present
        if os.path.isdir(folder):
            # delete it
            shutil.rmtree(folder)
            # report
            say("clean", f"removed {os.path.relpath(folder, ROOT)}/")


# step name -> function
STEPS = {"index": step_index, "prep": step_prep, "extra": step_extra, "pack": step_pack, "results": step_results,
         "evaluate": step_evaluate, "package": step_package, "clean": step_clean}


# command-line options
def build_arg_parser() -> argparse.ArgumentParser:
    # keep the help text's line breaks
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # steps to run, in the order given
    ap.add_argument("steps", nargs="+", choices=list(STEPS) + ["all"], help="steps to run")
    # WARC location
    ap.add_argument("--warc", default=DEFAULT_WARC, help="ClueWeb09 WARC file (default: 07.warc.gz next to build.py)")
    # student ID used in the zip name
    ap.add_argument("--id", default="team27", help="ID for the zip name <ID>_HW2.zip (default: team27)")
    # ignore the make-style up-to-date checks
    ap.add_argument("--force", action="store_true", help="rerun index, prep, extra and pack even if up to date")
    # parser
    return ap


# entry point
def main(argv=None) -> int:
    # parse the command line
    args = build_arg_parser().parse_args(argv)
    # resolve the WARC path against the current folder, so a relative --warc works from anywhere
    args.warc = os.path.abspath(args.warc)
    # expand "all" in place
    steps = []
    # every requested step
    for s in args.steps:
        # "all" means the whole pipeline
        steps += ALL_STEPS if s == "all" else [s]
    # start timer for the whole build
    t0 = time.time()
    # run the steps in order
    for s in steps:
        # each step stops the build itself on failure
        STEPS[s](args)
    # total time
    print(f"build finished in {time.time() - t0:.1f}s")
    # success
    return 0


# run as a script
if __name__ == "__main__":
    # exit with main's code
    sys.exit(main())
