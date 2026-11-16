import argparse
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
# query file used for every result file
QUERIES = "queries.txt"

# result file name -> extra search.py options (all runs use queries.txt)
RESULT_RUNS = [
    ("results.txt", []),
    ("results_bm25.txt", ["-w", "bm25"]),
    ("results_champions.txt", ["--champions"]),
    ("results_tiered.txt", ["--tiered"]),
    ("results_spell_correct.txt", ["--spell", "correct"]),
]

# top-level files copied into the submission zip
PACKAGE_FILES = [
    "README.md", "requirements.txt", "queries.txt", "build.py", "hw1_indexer.py", "prep.py",
    "hw2_index_prep.ipynb", "index_reader.py", "query_parser.py", "matcher.py", "ranker.py",
    "search.py", "spell.py",
]
# folders copied into the submission zip as a whole
PACKAGE_DIRS = ["nltk_data", "results", "output"]
# files every complete output/ folder must contain
REQUIRED_OUTPUT = [
    "dictionary.txt", "postings.txt", "offsets.tsv", "doc_norms.tsv", "doc_lengths.tsv", "doc_ids.tsv",
    "champions.tsv", "champions_bm25.tsv", "tiers.tsv", "tiers_bm25.tsv", "surface_vocab.tsv", "meta.json",
]

# steps run by "all", in order
ALL_STEPS = ["index", "prep", "results", "package"]

# text shown by --help
HELP_TEXT = """HW2 build script: rebuild the index from the WARC, preprocess it, regenerate results/, package the submission.
examples:
  python build.py all                     every step; steps whose output is already up to date are skipped
  python build.py index --warc PATH       HW1 index only, from a WARC somewhere else
  python build.py prep results            preprocessing and result files only
  python build.py package --id 112062500  write dist/112062500_HW2.zip
  python build.py all --force             rebuild everything even if it looks up to date
  python build.py clean                   delete output/ and dist/"""


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


# True when target exists and is at least as new as every source
def up_to_date(target: str, sources: list) -> bool:
    # a missing target always needs building
    if not os.path.exists(target):
        # build it
        return False
    # modification time of the target
    t = os.path.getmtime(target)
    # every existing source must be older
    return all(os.path.getmtime(s) <= t for s in sources if os.path.exists(s))


# step "index": HW1 indexer over the WARC -> output/dictionary.txt, output/postings.txt
def step_index(args: argparse.Namespace) -> None:
    # main output of the indexer
    postings = os.path.join(OUTPUT_DIR, "postings.txt")
    # the WARC is the only input
    if not os.path.exists(args.warc):
        # an existing index is still usable without the WARC
        if os.path.exists(postings) and not args.force:
            # keep it
            say("index", f"{args.warc} not found; keeping the existing index")
            # nothing to do
            return
        # cannot build at all
        say("index", f"{args.warc} not found; put 07.warc.gz next to build.py or pass --warc PATH")
        # stop the build
        sys.exit(1)
    # index newer than the WARC and the indexer: skip (make-style)
    if not args.force and up_to_date(postings, [args.warc, os.path.join(ROOT, "hw1_indexer.py")]):
        # report the skip
        say("index", "output/postings.txt is up to date (use --force to rebuild)")
        # nothing to do
        return
    # HW1's indexer loads NLTK stopwords at import time, so point NLTK at the bundled copy
    run_python("index", ["hw1_indexer.py", "-i", args.warc, "-o", OUTPUT_DIR, "--no-progress"],
               {"NLTK_DATA": NLTK_DIR})


# step "prep": prep.py -> offsets, norms, lengths, doc ids, champion lists, tiers, surface vocabulary, meta.json
def step_prep(args: argparse.Namespace) -> None:
    # prep.py needs the HW1 index
    if not os.path.exists(os.path.join(OUTPUT_DIR, "postings.txt")):
        # explain which step comes first
        say("prep", "output/postings.txt missing; run 'python build.py index' first")
        # stop the build
        sys.exit(1)
    # meta.json is written last by prep.py, so its time stamp marks a finished run
    meta = os.path.join(OUTPUT_DIR, "meta.json")
    # inputs whose change makes the preprocessing stale
    sources = [os.path.join(OUTPUT_DIR, "postings.txt"), os.path.join(ROOT, "prep.py")]
    # every output file present and newer than its inputs: skip
    complete = all(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in REQUIRED_OUTPUT)
    # make-style skip
    if not args.force and complete and up_to_date(meta, sources):
        # report the skip
        say("prep", "output/meta.json is up to date (use --force to rerun)")
        # nothing to do
        return
    # prep.py arguments: same index folder, WARC when available (N, TREC-IDs, surface vocabulary)
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


# step "results": every ranked list of results/ from queries.txt
def step_results(args: argparse.Namespace) -> None:
    # the search needs a finished preprocessing
    if not os.path.exists(os.path.join(OUTPUT_DIR, "meta.json")):
        # explain which step comes first
        say("results", "output/meta.json missing; run 'python build.py prep' first")
        # stop the build
        sys.exit(1)
    # make sure the folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # one search.py run per result file; they take seconds, so they always rerun
    for name, extra in RESULT_RUNS:
        # batch mode with the run's options
        run_python("results", ["search.py", "-f", QUERIES, "-o", os.path.join("results", name)] + extra)


# step "package": dist/<id>_HW2.zip with the code, results and the full output/ folder
def step_package(args: argparse.Namespace) -> None:
    # every output file must exist, or the TA cannot run search.py
    missing = [f for f in REQUIRED_OUTPUT if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    # incomplete preprocessing
    if missing:
        # list what is missing
        say("package", "output/ is incomplete (missing " + ", ".join(missing) + "); run 'python build.py prep' first")
        # stop the build
        sys.exit(1)
    # every result file must exist as well
    missing = [n for n, _ in RESULT_RUNS if not os.path.exists(os.path.join(RESULTS_DIR, n))]
    # incomplete results
    if missing:
        # list what is missing
        say("package", "results/ is incomplete (missing " + ", ".join(missing) + "); run 'python build.py results' first")
        # stop the build
        sys.exit(1)
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
            say("package", f"{name} not found")
            # stop the build
            sys.exit(1)
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
    say("package", f"done: {os.path.getsize(zip_path) / 2**20:.1f} MiB")


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
STEPS = {"index": step_index, "prep": step_prep, "results": step_results, "package": step_package,
         "clean": step_clean}


# command-line options
def build_arg_parser() -> argparse.ArgumentParser:
    # keep the examples' line breaks
    ap = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    # steps to run, in the order given
    ap.add_argument("steps", nargs="+", choices=list(STEPS) + ["all"], help="steps to run")
    # WARC location
    ap.add_argument("--warc", default=DEFAULT_WARC, help="ClueWeb09 WARC file (default: 07.warc.gz next to build.py)")
    # student ID used in the zip name
    ap.add_argument("--id", default="team27", help="ID for the zip name <ID>_HW2.zip (default: team27)")
    # ignore the make-style up-to-date checks
    ap.add_argument("--force", action="store_true", help="rebuild the index and rerun prep even if up to date")
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